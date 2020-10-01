import numpy as np
import torch
from torch import nn
import variable as var
import utils
torch.autograd.set_detect_anomaly(True)
https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
class A3C:
    def __init__(self, init_a2c):
        self.discount_factor = init_a2c["discount_factor"]
        self.state_dim = init_a2c["state_dim"]
        self.action_dim = init_a2c["action_space"]
        self.critic = Critic(init_a2c['critic']).cuda()
        self.actor = Actor(init_a2c['actor']).cuda()
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


class Actor:
    def __init__(self):


class Worker:
    def __init__(self):

