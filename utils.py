import torch
import variable as var
import numpy as np
from itertools import product
from functools import reduce
import operator
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


def to_tensor(a):
    return torch.tensor(a, device=var.device).float()


def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)


def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value


def rolling_window(a, window=3):

    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class TrainSession():
    def __init__(self, agents, env, seed):
        env.seed(seed)
        plt.style.use('ggplot')
        self.agents = agents
        self.env = env
        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
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
                set_in_dict(agent_init_tmp, name, value)
                agent_name += f"{'_'.join(name)}:{value};"

            agents.update({agent_name: agent_object(agent_init_tmp)})
            self.rewards_per_episode.update({agent_name: np.array([])})
            self.time_steps_per_episode.update({agent_name: np.array([])})

        self.agents.update(agents)

        return list(agents.keys())

    def plot_results(self, window=200, agent_subset=None, std=True):

        if not agent_subset:
            agent_subset = self.agents.keys()

        series_to_plot = {'rewards': {agent_name: self.rewards_per_episode[agent_name] for agent_name in agent_subset},
                          'time_steps': {agent_name: self.time_steps_per_episode[agent_name] for agent_name in agent_subset}}

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

                cm_idx = jdx % self.max_diff_colors # jdx // self.num_lines_style * float(self.num_lines_style) / self.max_diff_colors
                ls_idx = min(jdx // self.max_diff_colors, self.num_lines_style)  # jdx % self.num_lines_style

                series_mvg = rolling_window(series, window=window)
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

    def train(self, n_episode=500, t_max_per_episode=200, graphical=False, agent_subset=None):

        if agent_subset:
            agents = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        else:
            agents = self.agents

        for agent_name, agent in agents.items():

            time_steps_per_episode = list()
            rewards_per_episode = list()

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

                    rewards += reward

                    if done:
                        break

                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)

            self.time_steps_per_episode[agent_name] = np.concatenate([self.time_steps_per_episode[agent_name],
                                                                      np.array(time_steps_per_episode)])
            self.rewards_per_episode[agent_name] = np.concatenate([self.rewards_per_episode[agent_name],
                                                                   np.array(rewards_per_episode)])
