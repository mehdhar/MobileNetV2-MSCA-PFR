import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCA(nn.Module):
    """
    Multi-Scale Context Aggregator (MSCA)
    """

    def __init__(self, in_channels, out_channels):
        super(MSCA, self).__init__()

        # Parallel multi-scale branches
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4,
                                 kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.branch3 = nn.Conv2d(in_channels, out_channels // 4,
                                 kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        self.branch_dilated = nn.Conv2d(in_channels, out_channels // 4,
                                        kernel_size=3, stride=1, padding=3, dilation=3, bias=False)

        # Global-context pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, out_channels // 4,
                                     kernel_size=1, stride=1, bias=False)

        # Fuse all branches
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch3(x)
        b3 = self.branch_dilated(x)

        # Global pooling branch
        gp = self.global_pool(x)
        gp = self.global_conv(gp)
        gp = F.interpolate(gp, size=b3.shape[2:], mode="bilinear", align_corners=True)

        # Concatenate
        out = torch.cat([b1, b2, b3, gp], dim=1)

        # Fuse aggregated features
        out = self.fuse(out)
        out = self.bn(out)
        out = self.relu(out)

        return out
