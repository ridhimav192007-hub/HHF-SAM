import torch
import torch.nn as nn


class HFM(nn.Module):
    def __init__(self, channels=768):
        super().__init__()

        # Local information
        self.local_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

        # Regional information
        self.regional_conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=2, dilation=2
        )

        # Global information
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(
            channels, channels, kernel_size=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        local = self.relu(self.local_conv(x))
        regional = self.relu(self.regional_conv(x))

        global_feature = self.global_pool(x)
        global_feature = self.relu(
            self.global_conv(global_feature)
        )

        global_feature = global_feature.expand_as(x)

        output = local + regional + global_feature

        return output
    