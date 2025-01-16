import numpy as np
import torch
from torch import nn
from einops import rearrange


class DisparityDecoder(nn.Module):
    def __init__(self, c_in, c_out, heatmap_size, dmax, dmin, num_disp):
        super(DisparityDecoder, self).__init__()
        # in order to keep dmin to be at most 2.0
        self.offset = dmin - 1 if dmin > 2 else 0
        self.dmax = dmax - self.offset
        self.dmin = dmin - self.offset
        self.num_disp = num_disp

        self.disp_bin = self._gen_disp_bin()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(c_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=c_in // 2,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(
            (heatmap_size // 4) * (heatmap_size // 4), num_disp)

    def _gen_disp_bin(self):
        indices = np.arange(0, self.num_disp, 1)
        disparity = []
        for ind in indices:
            x = ind / (self.num_disp - 1) \
                * (np.log(self.dmax) - np.log(self.dmin)) \
                + np.log(self.dmin)
            d = np.exp(x) + self.offset
            disparity.append(d)

        # disparity = np.linspace(self.dmin, self.dmax, 50)
        disparity = nn.Parameter(
            torch.tensor(disparity).view(1, 1, -1).float(),
            requires_grad=False)

        return disparity

    def forward(self, x):
        x = self.conv(x)

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.linear(x)

        x = nn.functional.softmax(x, dim=2)
        x = torch.sum(self.disp_bin * x, dim=2, keepdim=True)

        return x
