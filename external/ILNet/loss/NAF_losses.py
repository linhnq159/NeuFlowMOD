# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision.models import vgg19
from loss.loss_util import weighted_loss
# from basicsr.models.archs.SPyNet import SPyNet, flow_warp

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss2(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss2, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # diff = torch.pow(torch.abs(pred - target), 2)
        diff = torch.sqrt((pred - target)**2 + 1e-12)
        if weight is not None:
            diff = diff*weight
            
        valid_pts = diff>0
        valid_pts = valid_pts.to(torch.int)
        if self.reduction == "mean":
            if weight is not None:
                return self.loss_weight*(diff.sum()/weight.sum())
            else:
                return self.loss_weight*(diff.sum()/valid_pts.sum())
        elif self.reduction == "sum":
            return self.loss_weight*diff.sum()
        else:
            return self.loss_weight*diff
        

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))

        return self.loss_weight * self.scale * torch.log(mse + 1.0).mean()

class MultiScaleTemporalLoss(nn.Module): 
    def __init__(self, loss_weight = 1.0):
        super(MultiScaleTemporalLoss, self).__init__()
        self.loss_weight = loss_weight
        self.C = [1,3,5,7]
        self.T = nn.ModuleList(
            nn.Sequential(nn.AvgPool2d(k, 1, int((k-1)/2)), nn.Softmax2d()) for k in self.C
        )
        
    def forward(self, pred, target):
        prev_loss = self.calc_loss(pred[:-1], target[:-1])
        post_loss = self.calc_loss(pred[1:], target[1:])
        return self.loss_weight*(prev_loss + post_loss)/2

    def calc_loss(self, pred, target):
        gt_errors, pred_errors = [], []
        # Error with prev frame:
        for t in self.T:
            gt_errors.append(t(target[1]) - t(target[0]))
            pred_errors.append(t(pred[1]) - t(pred[0]))
        
        gt_error_rgb_pixel_min = gt_errors[0]
        out_error_rgb_pixel_min = pred_errors[0]

        for j in range(1, len(self.C)):
            gt_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(pred_errors[j]),
                    gt_error_rgb_pixel_min, gt_errors[j])
            out_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(pred_errors[j]),
                    out_error_rgb_pixel_min, pred_errors[j])

        return F.l1_loss(gt_error_rgb_pixel_min, out_error_rgb_pixel_min)
    
    def calc_consistency(self, pred, target):
        gt_errors, pred_errors = [], []
        # Error with prev frame:
        for t in self.T:
            gt_errors.append(t(target[1]) - t(target[0]))
            pred_errors.append(t(pred[1]) - t(pred[0]))
        
        gt_error_rgb_pixel_min = gt_errors[0]
        
        out_error_rgb_pixel_min = pred_errors[0]
        
        for j in range(1, len(self.C)):
            gt_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(pred_errors[j]),
                    gt_error_rgb_pixel_min, gt_errors[j])
            out_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(pred_errors[j]),
                    out_error_rgb_pixel_min, pred_errors[j])

        gt_error_rgb_pixel_min = (gt_error_rgb_pixel_min - gt_error_rgb_pixel_min.min())/(gt_error_rgb_pixel_min.max() - gt_error_rgb_pixel_min.min())
        out_error_rgb_pixel_min = (out_error_rgb_pixel_min - out_error_rgb_pixel_min.min())/(out_error_rgb_pixel_min.max() - out_error_rgb_pixel_min.min())
        
        return 1 - F.l1_loss(out_error_rgb_pixel_min, gt_error_rgb_pixel_min)

class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight = 1.0, criterion = 'mse'):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = criterion
        
        vgg = vgg19()
        state_dict = torch.load("/data/chinnl/sources/image_enhancement/NAFNet/experiments/pretrained_models/vgg19-dcbb9e9d.pth", map_location="cuda:0")
        vgg.load_state_dict(state_dict)
        vgg.eval()

        self.vgg = vgg.features[:1]
        
    def forward(self, pred, target):
        self.vgg = self.vgg.to(pred.device)
         
        pred_ft = self.vgg(pred)
        target_ft = self.vgg(target)

        if self.criterion == 'mse':
            # loss = torch.pow(pred_ft - target_ft, 2).mean()
            loss = F.mse_loss(pred_ft, target_ft)
        elif self.criterion == 'l1':
            loss = F.l1_loss(pred_ft, target_ft)
        
        return loss*self.loss_weight, None # perceptual style is set to None
    
# class FlowBasedLoss(nn.Module):
#     def __init__(self, pretrained = "", loss_weight = 1.0, criterion = 'mse'):
#         super(FlowBasedLoss, self).__init__()
#         assert pretrained != "", ('SPyNet pretrained weights must not be None!')
#         self.loss_weight = loss_weight
#         self.criterion = criterion
        
#         self.spynet = SPyNet(pretrained = pretrained)
#         for name, param in self.spynet.named_parameters():
#             param.requires_grad = False
    
#     def forward(self, pred, target):
#         target1 = target.unsqueeze(0)[:, :-1, :, :].reshape(-1, 3, 640, 640)
#         target2 = target.unsqueeze(0)[:, 1:, :, :].reshape(-1, 3, 640, 640)
#         flow_target = self.spynet(target1, target2)
#         flow_target = (flow_target - flow_target.min())/(flow_target.max() - flow_target.min())
        
#         pred1 = pred.unsqueeze(0)[:, :-1, :, :].reshape(-1, 3, 640, 640)
#         pred2 = pred.unsqueeze(0)[:, 1:, :, :].reshape(-1, 3, 640, 640)
#         flow_pred = self.spynet(pred1, pred2)
#         flow_pred = (flow_pred - flow_pred.min())/(flow_pred.max() - flow_pred.min())
#         if self.criterion == 'mse':
#             loss = F.mse_loss(flow_pred, flow_target)
#         elif self.criterion == 'l1':
#             loss = F.l1_loss(flow_pred, flow_target)
        
#         return loss*self.loss_weight

# from basicsr.metrics.niqe import calculate_niqe

# class NIQELoss(nn.Module):
#     """
#     Custom NIQE Loss for unsupervised image enhancement.
#     """
#     def __init__(self, crop_border=0, input_order='HWC', convert_to='y'):
#         super(NIQELoss, self).__init__()
#         self.crop_border = crop_border
#         self.input_order = input_order
#         self.convert_to = convert_to

#     def forward(self, pred, target=None):
#         """
#         Args:
#             pred: Predicted images (batch_size, channels, height, width)
#             target: Optional target images (for paired training)
#         """
#         # For unsupervised training, we only use pred (enhanced images)
#         # Calculate NIQE score for each image in the batch
#         if pred.dim() == 4:  # batch_size, channels, height, width
#             batch_size = pred.shape[0]
#             niqe_scores = []
            
#             for i in range(batch_size):
#                 # Extract single image
#                 img = pred[i].cpu().detach().numpy()
                
#                 # Convert from CHW to HWC if needed
#                 if img.shape[0] == 3:
#                     img = np.transpose(img, (1, 2, 0))
                
#                 # Convert to uint8 if needed for NIQE calculation
#                 if img.dtype != np.uint8:
#                     img = (img * 255).clip(0, 255).astype(np.uint8)
                
#                 try:
#                     from basicsr.metrics.niqe import calculate_niqe
#                     niqe_score = calculate_niqe(img, self.crop_border, self.input_order, self.convert_to)
#                     niqe_scores.append(float(niqe_score))
#                 except Exception as e:
#                     print(f"Error calculating NIQE for image: {e}")
#                     niqe_scores.append(0.0)
            
#             # Return the mean NIQE score as loss (lower is better)
#             return torch.tensor(niqe_scores, dtype=torch.float32, device=pred.device).mean()
#         else:
#             # Handle single image
#             img = pred.cpu().detach().numpy()
#             if img.shape[0] == 3:
#                 img = np.transpose(img, (1, 2, 0))
            
#             if img.dtype != np.uint8:
#                 img = (img * 255).clip(0, 255).astype(np.uint8)
            
#             from basicsr.metrics.niqe import calculate_niqe
#             niqe_score = calculate_niqe(img, self.crop_border, self.input_order, self.convert_to)
#             return torch.tensor([float(niqe_score)], dtype=torch.float32, device=pred.device)