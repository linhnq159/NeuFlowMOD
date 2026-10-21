"""Single-class CenterNet head. Sizes/offsets are in stride-four feature units."""

import torch
from torch import nn
import torch.nn.functional as F

DETECTION_OUTPUTS = ('center_logits', 'size', 'offset')


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.center = nn.Conv2d(32, 1, 1)
        self.size = nn.Conv2d(32, 2, 1)
        self.offset = nn.Conv2d(32, 2, 1)
        nn.init.constant_(self.center.bias, -2.19)

    def forward(self, features):
        features = self.features(features)
        return dict(center_logits=self.center(features), size=F.softplus(self.size(features)) + 1e-6,
                    offset=torch.sigmoid(self.offset(features)))
