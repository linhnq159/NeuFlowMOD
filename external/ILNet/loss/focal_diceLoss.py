import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class FocalDiceLoss_Binary(nn.Module):
    def __init__(self, beta=2.0):
        super(FocalDiceLoss_Binary, self).__init__()
        self.beta = 1.0 / beta
    def _fdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        loss = 1 - loss**self.beta

        return loss

    def forward_single(self, inputs, target, weight=1.0, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        dice = self._fdice_loss(inputs[:, 0], target[:, 0])
        loss = dice * weight

        return loss 
    
    def forward(self, inputs, target, weight=1.0, sigmoid=True):

        total_loss = 0
        if isinstance(inputs, list):
            for item in inputs:
                total_loss += self.forward_single(item, target, weight, sigmoid)
        else:
            total_loss = self.forward_single(inputs, target, weight, sigmoid)




        return total_loss