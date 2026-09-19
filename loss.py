import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        # Binary cross-entropy loss
        bce_loss = self.bce(prediction, target)

        # Dice loss
        prediction = torch.sigmoid(prediction)

        intersection = (prediction * target).sum()

        dice = (2 * intersection + 1e-6) / (
            prediction.sum() + target.sum() + 1e-6
        )

        dice_loss = 1 - dice

        # Combined loss
        total_loss = bce_loss + dice_loss

        return total_loss
    