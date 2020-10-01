import numpy as np
from sklearn.preprocessing import OneHotEncoder
from oracle import OracleHandler
import variable as var
from collections import deque
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import os


class DnaRLEnv:

    def __init__(self, action_constraints, early_stop_reward, reward_func, seed=42, sequence_length=36,
                 reward_func_goal='mad', state_representation='latent', use_test_set=True, binary_reward_threshold=0.7):
        self.random_generator = np.random.RandomState(seed=seed)
        self.sequence_length = sequence_length
        self.reward_func_goal = reward_func_goal
        self.reward_func = reward_func
        self.binary_reward_threshold = binary_reward_threshold

        self.state_representation = state_representation
        self.state_dim = (400,) if state_representation == 'raw' else None
        self.action_dim = (3, sequence_length)

        self.nucleotides_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.nucleotides_mapping_reversed = {value: key for key, value in self.nucleotides_mapping.items()}
        self.optimization_sequence = None
        self.crossover_sequence = None
        self.best_episode_sequence = None
        self.one_hot_encoder = OneHotEncoder(sparse=False).fit(np.arange(len(self.nucleotides_mapping)).reshape(-1, 1))

        self.oracle = OracleHandler(checkpoint_path=var.oracle_checkpoint_path)
        self.action_constraints = self.load_action_constraints(action_constraints)
        self.early_stop_reward = EarlyStopReward(**early_stop_reward)
        self.test_set_length = None
        self.test_set_sequences = self.load_test_set_sequences() if use_test_set else iter([])
        self.oracle_scores = list()
        self.rewards = list()

        self.last_episode_score = 0.0
        self.best_episode_score = -1.0
        self.best_score_per_episode = list()
        self.best_sequence_per_episode = list()
        self.actions_history = list()

    def step(self, action):
        assert self.optimization_sequence is not None and self.crossover_sequence is not None, "sequences not initialized, use env.reset()"
        assert isinstance(action, (np.ndarray, )), "action must be a numpy.ndarray"
        assert action.shape[0] == self.action_dim[0], f'action must be a {self.action_dim[0]} array'

        co_length, opt_start_point, co_start_point = action

        assert 0 <= co_length < self.sequence_length, \
            f"co_length ({co_length}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= opt_start_point < self.sequence_length, \
            f"opt_start_point ({opt_start_point}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= co_start_point < self.sequence_length, \
            f"co_start_point ({co_start_point}) must be between a value between 0 and {self.sequence_length - 1}"

        action_verified = self.check_constraint(action)
        self.actions_history.append(action.tolist() + [action_verified])

        if not action_verified:
            return self.state(), self.reward_function(action_verified), self.early_stop_reward.has_to_stop()

        co_length = min(co_length, (self.sequence_length - 1) - max(opt_start_point, co_start_point))

        opt_subset = self.optimization_sequence[opt_start_point: opt_start_point + co_length].copy()
        co_subset = self.crossover_sequence[co_start_point: co_start_point + co_length].copy()

        self.optimization_sequence[opt_start_point: opt_start_point + co_length] = co_subset
        self.crossover_sequence[co_start_point: co_start_point + co_length] = opt_subset

        return self.state(), self.reward_function(action_verified), self.early_stop_reward.has_to_stop()

    def reset(self):

        if self.best_episode_sequence is not None and self.best_episode_score != 0.0:
            self.best_score_per_episode.append(self.best_episode_score)
            self.best_sequence_per_episode.append(self.best_episode_sequence)

        try:
            optimization_sequence = next(self.test_set_sequences)
            crossover_sequence = next(self.test_set_sequences)
        except StopIteration:
            optimization_sequence = self.random_generator.choice(list(self.nucleotides_mapping.values()),
                                                                 size=(self.sequence_length,))
            crossover_sequence = self.random_generator.choice(list(self.nucleotides_mapping.values()),
                                                              size=(self.sequence_length,))

        optimization_sequence = self.one_hot_sequence(optimization_sequence)
        crossover_sequence = self.one_hot_sequence(crossover_sequence)

        self.optimization_sequence = optimization_sequence
        self.crossover_sequence = crossover_sequence
        self.best_episode_sequence = deepcopy(optimization_sequence)

        self.best_episode_score = -1.0
        self.last_episode_score = 0.0

        state = self.state()

        if not self.state_dim:
            self.state_dim = state.shape

        return state

    def render(self, mode='pretty'):
        opt_sequence = None
        co_sequence = None

        if mode == 'pretty':
            opt_sequence = np.argmax(self.optimization_sequence, axis=1)
            co_sequence = np.argmax(self.crossover_sequence, axis=1)

            opt_sequence = np.vectorize(self.nucleotides_mapping_reversed.get)(opt_sequence)
            co_sequence = np.vectorize(self.nucleotides_mapping_reversed.get)(co_sequence)

        elif mode == 'raw':
            opt_sequence = self.optimization_sequence
            co_sequence = self.crossover_sequence

        print(f"opt seq\t:\t {opt_sequence}")
        print(f"co seq\t:\t {co_sequence}")

    def one_hot_sequence(self, sequence):
        return self.one_hot_encoder.transform(sequence.reshape(-1, 1))

    def state(self):
        state = None
        if self.state_representation == 'raw':
            state = np.concatenate((self.optimization_sequence.T, self.crossover_sequence.T), axis=1)
        if self.state_representation == 'latent':
            sequences = np.stack((self.optimization_sequence.T, self.crossover_sequence.T))
            embedded_sequences = self.oracle.get_latent(sequences)
            state = np.concatenate((embedded_sequences[0], embedded_sequences[1]), axis=0)
            # retrieve latent space of Oracle neural network for both opt and co, then concat.

        return state

    def reward_function(self, action_verified):
        score = -5.0
        if not action_verified:
            self.early_stop_reward.append_score(score)
            return score

        prediction = self.oracle.predict(self.optimization_sequence.T)

        if self.reward_func_goal == 'max':
            score = prediction[0].item()
        elif self.reward_func_goal == 'mad':
            score = prediction[1].item()

        self.oracle_scores.append(score)
        if score > self.best_episode_score:
            self.best_episode_score = score
            self.best_episode_sequence = deepcopy(self.optimization_sequence)

        if self.reward_func == 'relative_score_difference':
            reward = self.relative_score_difference_reward(score)
        elif self.reward_func == 'binary':
            reward = self.binary_reward(score)
        else:
            reward = 0.0

        self.early_stop_reward.append_score(score)
        self.rewards.append(reward)

        return reward

    def relative_score_difference_reward(self, score):

        reward = score - self.last_episode_score
        self.last_episode_score = score

        return reward

    def binary_reward(self, score):
        reward = -0.1
        if score >= self.early_stop_reward.reward_high_threshold:
            reward = 1.0
        return reward

    def get_state_dim(self):
        assert self.state_dim is not None, "state_dim not initialized. call env.reset() first"
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    @staticmethod
    def load_action_constraints(action_constraints):
        constraints = list()
        for action_name, (lower_bound, upper_bound) in action_constraints.items():
            constraints.append(ActionConstraint(lower_bound, upper_bound, action_name))

        return constraints

    def load_test_set_sequences(self):
        test_set_path = os.environ['PDBX_UBC'] + '/data/GA/Raluca/Combined_Max_Myc_Mad_Mad_r_log.xlsx'
        sequences = pd.read_excel(test_set_path, usecols=['Sequence'])['Sequence'].values
        self.random_generator.shuffle(sequences)
        self.test_set_length = len(sequences)
        sequences = [np.vectorize(self.nucleotides_mapping.get)(list(sequence)) for sequence in sequences]
        return iter(sequences)

    def check_constraint(self, actions):
        action_verified = True
        for action_constraint, action in zip(self.action_constraints, actions):
            action_verified = action_verified and action_constraint.check_constraint(action)

        return action_verified

    def seed(self, seed):
        self.random_generator.seed(seed)

    def cross_over_unitary_test(self, action, opt_seq, co_seq, next_opt_seq, next_co_seq):
        co_length, opt_start_point, co_start_point = action
        co_length = min(co_length, self.sequence_length - max(opt_start_point, co_start_point))

        opt_subset = opt_seq[opt_start_point: opt_start_point + co_length].copy()
        co_subset = co_seq[co_start_point: co_start_point + co_length].copy()

        opt_seq[opt_start_point: opt_start_point + co_length] = co_subset
        co_seq[co_start_point: co_start_point + co_length] = opt_subset

        return (next_opt_seq == opt_seq).all() and (next_co_seq == co_seq).all()

    def free_oracle_measures(self):
        self.best_score_per_episode.append(self.best_episode_score)
        self.best_sequence_per_episode.append(self.best_episode_sequence)

        best_score_per_episode = deepcopy(self.best_score_per_episode)
        scores = self.oracle_scores

        self.best_score_per_episode = list()
        self.oracle_scores = list()

        return scores, best_score_per_episode

    def free_actions_history(self):
        actions_history = deepcopy(self.actions_history)
        self.actions_history = list()

        return actions_history


class EarlyStopReward:
    def __init__(self, reward_low_threshold, reward_high_threshold, patience):
        self.rewards = deque(maxlen=patience)
        self.reward_high_threshold = reward_high_threshold
        self.reward_low_threshold = reward_low_threshold
        self.patience = patience

    def append_score(self, reward):
        self.rewards.append(reward)

    def has_to_stop(self):
        is_high = self.rewards[-1] >= self.reward_high_threshold
        is_low = False  # sum(self.rewards) / len(self.rewards) < self.reward_low_threshold

        return is_low or is_high


class ActionConstraint:
    def __init__(self, lower_constraint, upper_constraint, action_name):
        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint
        self.action_name = action_name

    def check_constraint(self, action):
        return self.lower_constraint < action < self.upper_constraint


class DnaRLEnvImmutableCoSeq(DnaRLEnv):
    def __init__(self, crossover_sequence, action_constraints, reward_func, early_stop_reward, seed=42, sequence_length=36, reward_func_goal='mad',
                 state_representation='latent', use_test_set=True, binary_reward_threshold=0.7):
        super(DnaRLEnvImmutableCoSeq, self).__init__(action_constraints=action_constraints,
                                                     reward_func=reward_func,
                                                     seed=seed,
                                                     reward_func_goal=reward_func_goal,
                                                     early_stop_reward=early_stop_reward,
                                                     use_test_set=False,
                                                     sequence_length=sequence_length,
                                                     state_representation=state_representation,
                                                     binary_reward_threshold=binary_reward_threshold)
        self.crossover_sequence = self.one_hot_sequence(crossover_sequence)

    def step(self, action):
        assert self.optimization_sequence is not None and self.crossover_sequence is not None, "sequences not initialized, use env.reset()"
        assert isinstance(action, (np.ndarray,)), "action must be a numpy.ndarray"
        assert action.shape[0] == self.action_dim[0], f'action must be a {self.action_dim[0]} array'

        co_length, opt_start_point, co_start_point = action

        assert 0 <= co_length < self.sequence_length, \
            f"co_length ({co_length}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= opt_start_point < self.sequence_length, \
            f"opt_start_point ({opt_start_point}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= co_start_point < self.sequence_length, \
            f"co_start_point ({co_start_point}) must be between a value between 0 and {self.sequence_length - 1}"

        action_verified = self.check_constraint(action)
        self.actions_history.append(action.tolist() + [action_verified])

        if not action_verified:
            return self.state(), self.reward_function(action_verified), self.early_stop_reward.has_to_stop()

        co_length = min(co_length, (self.sequence_length - 1) - max(opt_start_point, co_start_point))

        co_subset = self.crossover_sequence[co_start_point: co_start_point + co_length].copy()

        self.optimization_sequence[opt_start_point: opt_start_point + co_length] = co_subset

        return self.state(), self.reward_function(action_verified), self.early_stop_reward.has_to_stop()

    def reset(self):
        if self.best_episode_sequence is not None and self.best_episode_score != 0.0:
            self.best_score_per_episode.append(self.best_episode_score)
            self.best_sequence_per_episode.append(self.best_episode_sequence)

        try:
            optimization_sequence = next(self.test_set_sequences)
        except StopIteration:
            optimization_sequence = self.random_generator.choice(list(self.nucleotides_mapping.values()),
                                                                 size=(self.sequence_length,))

        optimization_sequence = self.one_hot_sequence(optimization_sequence)

        self.optimization_sequence = optimization_sequence
        self.best_episode_sequence = deepcopy(optimization_sequence)

        self.best_episode_score = -1.0
        self.last_episode_score = 0.0

        state = self.state()

        if not self.state_dim:
            self.state_dim = state.shape

        return state

    def state(self):
        state = None
        if self.state_representation == 'raw':
            state = self.optimization_sequence
        if self.state_representation == 'latent':
            sequence = np.expand_dims(self.optimization_sequence.T, axis=0)
            embedded_sequence = self.oracle.get_latent(sequence)
            state = embedded_sequence[0]
            # retrieve latent space of Oracle neural network for both opt and co, then concat.

        return state



