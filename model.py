import torch.nn as nn
import torch.nn.functional as F

class ChipConv(nn.Module):
    def __init__(self, mch):
        super(ChipConv, self).__init__()

        self.conv = nn.Conv2d(mch, mch, (3, 3), padding=1)
        self.dilconv = nn.Conv2d(mch, mch, (3, 3), dilation=2, padding=2)
        self.bn = nn.BatchNorm2d(mch)

    def forward(self, x):
        y = x + self.conv(x) + self.dilconv(x)
        return F.relu(self.bn(y))


class ChipNet(nn.Module):
    def __init__(self, ich=14, mch=64, och=2, n_layers=10):
        super(ChipNet, self).__init__()

        self.conv_in = nn.Conv2d(ich, mch, (5, 5), padding=2)
        self.bn = nn.BatchNorm2d(mch)
        self.conv_chip = nn.Sequential(*[ChipConv(mch) for _ in range(n_layers)])
        self.conv_out = nn.Conv2d(mch, och, (1, 1))

    def forward(self, x):
        y = F.relu(self.bn(self.conv_in(x)))
        y = F.relu(self.conv_chip(y))
        y = F.relu(self.conv_out(y))
        return y