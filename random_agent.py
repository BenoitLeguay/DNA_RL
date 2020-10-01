import numpy as np


class RandomAgent:
    def __init__(self, seed=42):
        self.random_state = np.random.RandomState(seed=seed)

    def policy(self):
        return self.random_state.choice(range(36), size=(3,))

    def episode_init(self, state):
        return self.policy()

    def update(self, state, reward, done):
        return self.policy()
