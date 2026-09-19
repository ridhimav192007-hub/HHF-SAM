import torch
import torch.nn as nn
import torch.nn.functional as F


class PHFD(nn.Module):
    def __init__(self, channels=768):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, 256, kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            256, 128, kernel_size=3, padding=1
        )

        self.conv3 = nn.Conv2d(
            128, 64, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()

        self.mask_head = nn.Conv2d(
            64, 1, kernel_size=1
        )

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False
        )

        x = self.relu(self.conv2(x))
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False
        )

        x = self.relu(self.conv3(x))
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False
        )

        mask = self.mask_head(x)

        return mask
    