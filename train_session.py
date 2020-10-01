import torch
import variable as var
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import os
import utils as ut
import pandas as pd


class TrainSession:
    def __init__(self, agents, env, seed):
        env.seed(seed)
        plt.style.use('ggplot')
        self.agents = agents
        self.env = env

        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.reward_per_time_step = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.best_score_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.score_per_time_step = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.actions_history = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        self.num_lines_style = len(self.line_styles)
        self.cm = plt.get_cmap('tab10')
        self.max_diff_colors = 8

    def append_agents(self, agents, overwrite=False):

        assert not any(item in agents for item in self.agents) or overwrite, "You are trying to overwrite agents dictionary"
        agent_names = list(agents.keys())

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.reward_per_time_step.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.best_score_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.score_per_time_step.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.actions_history.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

        return agent_names

    def pop_agents(self, agents):
        valid_agent_name = set(agents).intersection(self.agents.keys())
        for agent_name in valid_agent_name:
            self.agents.pop(agent_name)

    def parameter_grid_append(self, agent_object, base_agent_init, parameters_dict):

        agents = {}
        parameter_grid = list(dict(zip(parameters_dict, x)) for x in product(*parameters_dict.values()))
        for parameters_dict in parameter_grid:
            agent_init_tmp = deepcopy(base_agent_init)
            agent_name = ""
            for name, value in parameters_dict.items():
                ut.set_in_dict(agent_init_tmp, name, value)
                agent_name += f"{'_'.join(name)}:{value};"

            agents.update({agent_name: agent_object(agent_init_tmp)})
            self.rewards_per_episode.update({agent_name: np.array([])})
            self.time_steps_per_episode.update({agent_name: np.array([])})
            self.reward_per_time_step.update({agent_name: np.array([])})
            self.best_score_per_episode.update({agent_name: np.array([])})
            self.score_per_time_step.update({agent_name: np.array([])})
            self.actions_history.update({agent_name: np.array([])})

        self.agents.update(agents)

        return list(agents.keys())

    def plot_results(self, window=200, agent_subset=None, std=True):

        if not agent_subset:
            agent_subset = self.agents.keys()

        series_to_plot = {'cumulative rewards': {agent_name: self.rewards_per_episode[agent_name] for agent_name in agent_subset},
                          'best score': {agent_name: self.best_score_per_episode[agent_name] for agent_name in agent_subset},
                          'time steps': {agent_name: self.time_steps_per_episode[agent_name] for agent_name in agent_subset}
                          }

        agents_to_plot = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        loss_per_agents = {'critic_loss': {agent_name: (np.array(agent.critic.loss_history) if 'critic' in agent.__dict__.keys()
                                                 else np.array([]))
                                    for agent_name, agent
                                    in agents_to_plot.items()},
                           'actor_loss': {agent_name: (np.array(agent.actor.loss_history) if 'actor' in agent.__dict__.keys()
                                                       else np.array([]))
                                          for agent_name, agent
                                          in agents_to_plot.items()}
                           }

        series_to_plot.update(loss_per_agents)

        fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 20), facecolor='w', edgecolor='k')
        axs = axs.ravel()

        for idx, (series_name, dict_series) in enumerate(series_to_plot.items()):
            for jdx, (agent_name, series) in enumerate(dict_series.items()):
                if series.size == 0:
                    axs[idx].plot([0.0], [0.0], label=agent_name)
                    continue

                cm_idx = jdx % self.max_diff_colors
                # jdx // self.num_lines_style * float(self.num_lines_style) / self.max_diff_colors (upward)
                ls_idx = min(jdx // self.max_diff_colors, self.num_lines_style)  # jdx % self.num_lines_style

                series_mvg = ut.rolling_window(series, window=window)
                series_mvg_avg = np.mean(series_mvg, axis=1)

                lines = axs[idx].plot(range(len(series_mvg_avg)), series_mvg_avg, label=agent_name)

                lines[0].set_color(self.cm(cm_idx))
                lines[0].set_linestyle(self.line_styles[ls_idx])

                if std:
                    series_mvg_std = np.std(series_mvg, axis=1)
                    area = axs[idx].fill_between(range(len(series_mvg_avg)), series_mvg_avg - series_mvg_std,
                                                 series_mvg_avg + series_mvg_std, alpha=0.15)
                    area.set_color(self.cm(cm_idx))
                    area.set_linestyle(self.line_styles[ls_idx])

            box = axs[idx].get_position()
            axs[idx].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[idx].set_title(f"{series_name} per episode", fontsize=15)
            axs[idx].set_ylabel(f"avg {series_name}", fontsize=10)
            axs[idx].set_xlabel(f"episodes", fontsize=10)
            axs[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()

    def plot_density(self):
        scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.score_per_time_step.items()]))
        best_scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.best_score_per_episode.items()]))

        test_set = pd.read_csv("test_set.csv", index_col=0)
        test_set_scores = test_set.loc[:, self.env.reward_func_goal].rename('Test set score')

        pd.concat([scores, test_set_scores], axis=1).plot.kde(bw_method=.1, title='scores')
        pd.concat([best_scores, test_set_scores], axis=1).plot.kde(bw_method=.1, title='Best scores/ ep')

    def plot_actions_distributions(self):

        act_con = self.env.action_constraints
        columns = [f'Cross-over Length {act_con[0].lower_constraint} < l < {act_con[0].upper_constraint}',
                   f'Index Optimization sequence {act_con[1].lower_constraint} < Iopt < {act_con[1].upper_constraint}',
                   f'Index Cross-over sequence {act_con[2].lower_constraint} < Ico < {act_con[2].upper_constraint}',
                   'Validated actions']
        for agent_name, actions_history in self.actions_history.items():

            actions_history = pd.DataFrame(actions_history, columns=columns)
            fig, axes = plt.subplots(2, 2, figsize=(16, 8))

            for idx, col in enumerate(actions_history.columns):
                unravel_idx = np.unravel_index(idx, axes.shape)
                vc = actions_history[col].value_counts(sort=False)
                vc.reindex(range(36), fill_value=0).plot(kind='bar', title=col, ax=axes[unravel_idx]) if idx != 3 else \
                    vc.sort_index().plot(kind='bar', title=col, ax=axes[unravel_idx])

                axes[unravel_idx].axvspan(act_con[idx].lower_constraint + .75,
                                          act_con[idx].upper_constraint - .75,
                                          alpha=0.25, color='green') if idx != 3 else None
            fig.suptitle(agent_name.upper())
            plt.show()

    def train(self, n_episode=500, t_max_per_episode=200, graphical=False, agent_subset=None):

        if agent_subset:
            agents = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        else:
            agents = self.agents

        for agent_name, agent in agents.items():

            time_steps_per_episode = list()
            rewards_per_episode = list()
            reward_per_time_step = list()

            for _ in tqdm(range(n_episode)):

                rewards = 0.0
                state = self.env.reset()
                next_action = agent.episode_init(state)
                for t in range(t_max_per_episode):
                    if graphical:
                        self.env.render()

                    state, reward, done = self.env.step(
                        next_action)  # problem when the for loop end, while done is not True (agent_end not called)
                    next_action = agent.update(state, reward, done)

                    reward_per_time_step.append(reward)
                    rewards += reward

                    if done:
                        break

                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)

            scores, best_score_per_episode = self.env.free_oracle_measures()
            actions_history = self.env.free_actions_history()

            self.time_steps_per_episode[agent_name] = np.concatenate([self.time_steps_per_episode[agent_name],
                                                                      np.array(time_steps_per_episode)])
            self.rewards_per_episode[agent_name] = np.concatenate([self.rewards_per_episode[agent_name],
                                                                   np.array(rewards_per_episode)])
            self.reward_per_time_step[agent_name] = np.concatenate([self.reward_per_time_step[agent_name],
                                                                    np.array(reward_per_time_step)])
            self.best_score_per_episode[agent_name] = np.concatenate([self.best_score_per_episode[agent_name],
                                                                      np.array(best_score_per_episode)])
            self.score_per_time_step[agent_name] = np.concatenate([self.score_per_time_step[agent_name],
                                                                   np.array(scores)])
            self.actions_history[agent_name] = np.concatenate([self.actions_history[agent_name],
                                                               np.array(actions_history)]) \
                if self.actions_history[agent_name].size else np.array(actions_history)

    def save_model(self, suffix=''):
        model_dir = os.path.join(var.PATH, 'saved_model/')
        for agent_name, agent in self.agents.items():
            torch.save(agent.critic.state_dict(), os.path.join(model_dir, f"{agent_name}_critic_{suffix}.pth"))
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"{agent_name}_actor_{suffix}.pth"))

    def load_model(self, agent_name, suffix=''):
        model_dir = os.path.join(var.PATH, 'saved_model/')
        self.agents[agent_name].critic.load_state_dict(
            torch.load(os.path.join(model_dir, f"{agent_name}_critic_{suffix}.pth"))
        )
        self.agents[agent_name].actor.load_state_dict(
            torch.load(os.path.join(model_dir, f"{agent_name}_actor_{suffix}.pth"))
        )
