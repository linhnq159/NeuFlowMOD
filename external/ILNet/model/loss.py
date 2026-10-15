import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):

    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss

def dice_loss(pred, target, smooth=1.):
    """Dice loss
    """
    pred = pred.contiguous()
    pred = F.sigmoid(pred)
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    temp = intersection.sum()
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def dice_critetion(inputs, target):
    if isinstance(inputs, list):
        losses = [dice_loss(inputs[i], target) for i in range(len(inputs))]
        # total_loss = sum(losses)/len(inputs)
        total_loss = sum(losses)
    else:
        total_loss = dice_loss(inputs, target)

    return total_loss
    
def criterion(inputs, target):
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)

    return total_loss