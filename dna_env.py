import numpy as np
from sklearn.preprocessing import OneHotEncoder
from oracle import OracleHandler
import variable as var


class DnaRLEnv:

    def __init__(self, sequence_length=36):
        self.sequence_length = sequence_length
        self.nucleotides_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.nucleotides_mapping_reversed = {value: key for key, value in self.nucleotides_mapping.items()}
        self.optimization_sequence = None
        self.crossover_sequence = None
        self.one_hot_encoder = OneHotEncoder(sparse=False).fit(np.arange(4).reshape(-1, 1))
        self.oracle = OracleHandler(checkpoint_path=var.oracle_checkpoint_path)

    def step(self, action):
        assert self.optimization_sequence is not None and self.crossover_sequence is not None, "sequences not initialized, use env.reset()"
        assert isinstance(action, (np.ndarray, )), "action must be a numpy.ndarray"

        co_length, opt_start_point, co_start_point = action

        assert 0 <= co_length < self.sequence_length, \
            f"co_length ({co_length}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= opt_start_point < self.sequence_length, \
            f"opt_start_point ({opt_start_point}) must be between a value between 0 and {self.sequence_length - 1}"
        assert 0 <= co_start_point < self.sequence_length, \
            f"co_start_point ({co_start_point}) must be between a value between 0 and {self.sequence_length - 1}"

        co_length = min(co_length, (self.sequence_length - 1) - max(opt_start_point, co_start_point))

        opt_subset = self.optimization_sequence[opt_start_point: opt_start_point + co_length].copy()
        co_subset = self.crossover_sequence[co_start_point: co_start_point + co_length].copy()

        self.optimization_sequence[opt_start_point: opt_start_point + co_length] = co_subset
        self.crossover_sequence[co_start_point: co_start_point + co_length] = opt_subset

        return self.state(), self.reward_function()

    def reset(self):
        optimization_sequence = np.random.choice(list(self.nucleotides_mapping.values()), size=(self.sequence_length,))
        crossover_sequence = np.random.choice(list(self.nucleotides_mapping.values()), size=(self.sequence_length,))

        optimization_sequence = self.one_hot_sequence(optimization_sequence)
        crossover_sequence = self.one_hot_sequence(crossover_sequence)

        self.optimization_sequence = optimization_sequence
        self.crossover_sequence = crossover_sequence

        return self.state()

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

    def state(self, mode='raw'):
        if mode == 'raw':
            np.concatenate((self.optimization_sequence.T, self.crossover_sequence.T), axis=1)
        if mode == 'latent':
            pass  # retrieve latent space of Oracle neural network for both opt and co, then concat.

    def reward_function(self, mode='mad'):
        prediction = self.oracle.predict(self.optimization_sequence.T)
        score = 0.0
        if mode == 'max':
            score = prediction[0].item()
        if mode == 'mad':
            score = prediction[1].item()

        return score

