import numpy as np
import torch
from torch import nn
import variable as var
import utils


class A2C:
    def __init__(self, init_a2c):
        self.discount_factor = init_a2c["discount_factor"]
        self.state_dim = init_a2c["state_dim"]
        self.action_dim = init_a2c["action_space"]
        self.critic = Critic(init_a2c['critic']).cuda()
        self.actor = var.actor_types[init_a2c['actor_type']](init_a2c['actor']).cuda()
        self.optim_actor = None
        self.random_generator = np.random.RandomState(seed=init_a2c['seed'])
        self.next_state = None
        self.next_action = None

        self.init_optimizers(critic_optimizer=init_a2c['critic']['optimizer'],
                             actor_optimizer=init_a2c['actor']['optimizer'])

    def init_optimizers(self, critic_optimizer={}, actor_optimizer={}):
        self.critic.init_optimizer(critic_optimizer)
        self.actor.init_optimizer(actor_optimizer)

    def policy(self, state):
        with torch.no_grad():
            action = self.actor.predict_action(state)
        return action

    def episode_init(self, state):
        state = utils.to_tensor(state).view((1, ) + state.shape)

        action = self.policy(state)
        self.next_action = action
        self.next_state = state

        return action.cpu().numpy()

    def update(self, state, reward, done):

        state = utils.to_tensor(state).view((1, ) + state.shape)
        next_action = -1
        if not done:
            next_action = self.update_step(state, reward)
        if done:
            self.update_end(reward)

        return next_action

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        next_state_value = self.critic.estimate_state(next_state)
        current_state_value = self.critic.estimate_state(current_state)

        td_target = reward + self.discount_factor * next_state_value
        td_error = td_target - current_state_value

        self.actor.update(current_state, current_action, td_error)
        self.critic.update(current_state_value, td_target)

        next_action = self.policy(next_state)

        self.next_state = next_state
        self.next_action = next_action

        return next_action.cpu().numpy()

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        current_state_value = self.critic.estimate_state(current_state)
        td_target = utils.to_tensor([[float(reward)]])
        td_error = td_target - current_state_value

        self.actor.update(current_state, current_action, td_error)
        self.critic.update(current_state_value, td_target)


class Critic(torch.nn.Module):
    def __init__(self, critic_init):
        super(Critic, self).__init__()
        network_init = critic_init['network']
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Sequential(
            nn.ConstantPad1d(15 // 2, 0.25),
            nn.Conv1d(4, 400, 15),
            nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool1d(1))

        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.o = nn.Linear(network_init["l2_size"], 1)

        self.optimizer = None
        self.loss = torch.nn.MSELoss()
        self.loss_history = list()
        self.state_representation = critic_init['state_representation']

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):

        if self.state_representation == 'raw':
            x = self.conv1(x).squeeze(dim=2)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.o(x)

        return x

    def estimate_state(self, state):
        return self(state)

    def update(self, current_state_value, td_target):

        loss = self.loss(current_state_value, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class Actor(torch.nn.Module):
    def __init__(self, actor_init):
        super(Actor, self).__init__()
        torch.manual_seed(actor_init['seed'])
        network_init = actor_init['network']
        self.action_dim = actor_init['action_dim']
        self.entropy_learning_rate = actor_init['entropy_learning_rate']
        self.optimizer = None
        self.loss_history = list()
        self.state_representation = actor_init["state_representation"]

        self.hidden_size = network_init["hidden_size"]
        self.l1_size = network_init["l1_size"]

        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.ConstantPad1d(15 // 2, 0.25),
            nn.Conv1d(4, 400, 15),
            nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool1d(1))

        self.l1 = nn.Linear(network_init["i_size"] + self.hidden_size, self.l1_size)
        self.l1_to_h = nn.Linear(self.l1_size, self.hidden_size)
        self.l1_to_o = nn.Linear(self.l1_size, network_init["o_size"])
        self.softmax = nn.Softmax(dim=1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x, hidden):

        if self.state_representation == 'raw':
            x = self.conv1(x).squeeze(dim=2)

        combined = torch.cat((x, hidden), 1)

        l1_out = self.relu(self.l1(combined))
        hidden = self.relu(self.l1_to_h(l1_out))
        output = self.relu(self.l1_to_o(l1_out))
        output = self.softmax(output) #  .clone()

        return output, hidden

    def rnn_forward(self, x):
        #  hidden = torch.zeros(1, self.hidden_size, device=var.device) slower than empty and then fill
        hidden = torch.empty(x.shape[0], self.hidden_size, device=var.device).fill_(0)
        outputs = list()

        for _ in range(self.action_dim):
            output, hidden = self(x, hidden)
            outputs.append(output)

        return torch.stack(outputs, dim=0).squeeze(dim=1)

    def predict_action(self, state):  # return an action

        action_probabilities = self.rnn_forward(state)
        action_distributions = torch.distributions.Categorical(probs=action_probabilities)

        return action_distributions.sample()

    def update(self, state, action, td_error):

        actions_probabilities = self.rnn_forward(state)
        action_chosen_prob = torch.gather(actions_probabilities, dim=1, index=action.unsqueeze(dim=1))

        sum_entropy = torch.distributions.Categorical(probs=actions_probabilities).entropy().sum()

        loss = -torch.log(action_chosen_prob.prod()) * td_error - self.entropy_learning_rate * sum_entropy

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ActorVanilla:
    def __init__(self, actor_vanilla_init):
        self.action_names = ['co_length', 'opt_start_point', 'co_start_point']
        self.actors = {action_name: OneActionActor(actor_vanilla_init[action_name])
                       for action_name in self.action_names}
        self.loss_history = list()

    def cuda(self):
        for actor_name, actor in self.actors.items():
            self.actors[actor_name] = actor.cuda()
        return self

    def init_optimizer(self, optimizer_args):
        for actor_name, actor in self.actors.items():
            actor.init_optimizer(optimizer_args=optimizer_args[actor_name])

    def predict_action(self, state):
        actions_chosen = list()
        for actor_name, actor in self.actors.items():
            actions_chosen.append(actor.predict_action(state))

        return utils.to_tensor(actions_chosen).long()

    def update(self, state, action, td_error):
        loss_value = 0.0
        for idx, (actor_name, actor) in enumerate(self.actors.items()):
            loss_value += actor.update(state.clone(), action[idx].clone(), td_error.clone())

        self.loss_history.append(loss_value)


class OneActionActor(torch.nn.Module):
    def __init__(self, one_action_actor_init):
        network_init = one_action_actor_init['network']
        super(OneActionActor, self).__init__()
        torch.manual_seed(one_action_actor_init['seed'])
        self.entropy_learning_rate = one_action_actor_init['entropy_learning_rate']
        self.optimizer = None
        self.loss_history = list()
        self.state_representation = one_action_actor_init["state_representation"]

        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.ConstantPad1d(15 // 2, 0.25),
            nn.Conv1d(4, 400, 15),
            nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool1d(1))

        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.l3 = nn.Linear(network_init["l2_size"], network_init["o_size"])
        self.softmax = nn.Softmax(dim=1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):

        if self.state_representation == 'raw':
            x = self.conv1(x).squeeze(dim=2)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))

        return x

    def predict_action(self, state):  # return an action
        action_probabilities = self(state)
        action_distributions = torch.distributions.Categorical(probs=action_probabilities)

        return action_distributions.sample()

    def update(self, state, action, td_error):

        actions_probabilities = self(state)
        action_chosen_prob = torch.gather(actions_probabilities.squeeze(), dim=0, index=action)

        sum_entropy = torch.distributions.Categorical(probs=actions_probabilities).entropy().sum()

        loss = -torch.log(action_chosen_prob.prod()) * td_error - self.entropy_learning_rate * sum_entropy

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return loss_value


class NActionActor(torch.nn.Module):
    def __init__(self, one_action_actor_init):
        network_init = one_action_actor_init['network']
        super(NActionActor, self).__init__()
        torch.manual_seed(one_action_actor_init['seed'])
        self.action_dim = network_init["o_size"]//3
        self.entropy_learning_rate = one_action_actor_init['entropy_learning_rate']
        self.optimizer = None
        self.loss_history = list()
        self.state_representation = one_action_actor_init["state_representation"]

        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.ConstantPad1d(15 // 2, 0.25),
            nn.Conv1d(4, 400, 15),
            nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool1d(1))

        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.l3 = nn.Linear(network_init["l2_size"], network_init["o_size"])
        self.softmax = nn.Softmax(dim=1)
        self.softmax_dim_0 = nn.Softmax(dim=0)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):

        if self.state_representation == 'raw':
            x = self.conv1(x).squeeze(dim=2)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))

        return x[0]

    def predict_action(self, state):  # return an action
        action_values = self(state)
        actions_chosen = list()

        for idx in range(0, self.action_dim * 3, self.action_dim):
            probabilities = self.softmax_dim_0(action_values[idx:idx + self.action_dim])
            dis = torch.distributions.Categorical(probs=probabilities)
            actions_chosen.append(dis.sample())

        return utils.to_tensor(actions_chosen).long()

    def update(self, state, action, td_error):

        action_values = self(state)
        probabilities = list()

        for idx in range(0, self.action_dim * 3, self.action_dim):
            prob = self.softmax_dim_0(action_values[idx:idx + self.action_dim])
            probabilities.append(prob)

        probabilities = torch.stack(probabilities)

        action_chosen_prob = torch.gather(probabilities, dim=1, index=action.unsqueeze(dim=1))

        sum_entropy = torch.distributions.Categorical(probs=probabilities).entropy().sum()

        loss = -torch.log(action_chosen_prob.prod()) * td_error - self.entropy_learning_rate * sum_entropy

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return loss_value
