import torch
from LitNet import LitNet
import utils


class OracleHandler:
    def __init__(self, checkpoint_path):
        self.oracle = LitNet.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        if torch.cuda.is_available():
            self.oracle.cuda()

    def predict(self, sequence):
        with torch.no_grad():
            sequence = utils.to_tensor(sequence).view((1,) + sequence.shape)
            score = self.oracle(sequence)

        return score.squeeze().cpu().numpy()


