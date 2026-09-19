import torch
import torch.nn as nn


class MHPM(nn.Module):
    def __init__(self, channels=768):
        super().__init__()

        # Multi-scale convolution branches
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2
        )

        self.conv3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=4, dilation=4
        )

        self.relu = nn.ReLU()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))

        # Combine multi-scale features
        out = x1 + x2 + x3

        # Channel attention
        attention = self.avg_pool(out)
        attention = attention.view(attention.size(0), -1)
        attention = self.channel_attention(attention)
        attention = attention.unsqueeze(-1).unsqueeze(-1)

        out = out * attention

        return out
    



