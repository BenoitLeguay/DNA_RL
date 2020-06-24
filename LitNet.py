import pytorch_lightning as pl
import torch.nn as nn
import torch


class LitNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        num_filters_c1 = 100
        out_fc = 200

        self.cpa1 = nn.ConstantPad1d(15 // 2, 0.25)
        self.c1 = nn.Conv1d(4, num_filters_c1, 15)
        self.llr = nn.LeakyReLU(0.1)
        self.mp1 = nn.AdaptiveMaxPool1d(1)
        self.ap1 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(num_filters_c1 * 2, out_fc)
        self.fc2 = nn.Linear(out_fc, out_fc)
        self.fcf = nn.Linear(out_fc, len(self.hparams.column_names))
        self.fa = nn.Sigmoid()

    def forward(self, x):
        x = self.cpa1(x)
        x = self.llr(self.c1(x))
        x1 = self.mp1(x)
        x2 = self.ap1(x)
        x = torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)  # concat along last dimension
        x = self.llr(self.fc1(x))
        x = self.llr(self.fc2(x))
        x = self.fa(self.fcf(x))
        return x
