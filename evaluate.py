import torch


# Dice Score
def dice_score(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    intersection = (prediction * target).sum()

    dice = (
        2 * intersection + 1e-6
    ) / (
        prediction.sum() + target.sum() + 1e-6
    )

    return dice.item()


# IoU Score
def iou_score(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    intersection = (prediction * target).sum()

    union = prediction.sum() + target.sum() - intersection

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return iou.item()