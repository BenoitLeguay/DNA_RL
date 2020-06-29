import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import variable as var
import utils
torch.autograd.set_detect_anomaly(True)


class A2C:
    def __init__(self, init_a2c):
        self.discount_factor = init_a2c["discount_factor"]
        self.state_dim = init_a2c["state_dim"]
        self.action_dim = init_a2c["action_space"]
        self.critic = Critic(init_a2c['critic'], init_a2c['critic']['network']).cuda()
        self.actor = Actor(init_a2c['actor'], init_a2c['actor']['network']).cuda()
        self.optim_actor = None
        self.random_generator = np.random.RandomState(seed=init_a2c['seed'])
        self.next_state = None
        self.next_action = None

        self.init_optimizers()

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
    def __init__(self, critic_init, network_init):
        super(Critic, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.l3 = nn.Linear(network_init["l2_size"], network_init["l3_size"])
        self.l4 = nn.Linear(network_init["l3_size"], 1)

        self.optimizer = None
        self.loss = torch.nn.MSELoss()
        self.loss_history = list()

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)

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
    def __init__(self, actor_init, network_init):
        super(Actor, self).__init__()
        torch.manual_seed(actor_init['seed'])
        self.action_dim = actor_init['action_dim']
        self.entropy_learning_rate = actor_init['entropy_learning_rate']
        self.optimizer = None
        self.loss_history = list()

        self.hidden_size = network_init["hidden_size"]
        self.l1_size = network_init["l1_size"]

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(network_init["i_size"] + self.hidden_size, self.l1_size)
        self.l1_to_h = nn.Linear(self.l1_size, self.hidden_size)
        self.l1_to_o = nn.Linear(self.l1_size, network_init["o_size"])
        self.softmax = nn.Softmax(dim=1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x, hidden):

        combined = torch.cat((x, hidden), 1)
        l1_out = self.relu(self.l1(combined))
        hidden = self.relu(self.l1_to_h(l1_out))
        output = self.relu(self.l1_to_o(l1_out))
        output = self.softmax(output).clone()

        return output, hidden

    def rnn_forward(self, x):
        #  hidden = torch.zeros(1, self.hidden_size, device=var.device) slower than empty and then fill
        hidden = torch.empty(1, self.hidden_size, device=var.device).fill_(0)
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
