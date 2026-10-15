# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuFlow.neuflow import NeuFlow
from external.ILNet.model.LW_Opticalflow import LW_OpticalFLow
import torchvision.transforms as transforms

### normalize optical flow



class ModelBuilder(nn.Module):
    """Fine-tune NeuFlow through mask segmentation loss; flow labels are not required."""

    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone_op = NeuFlow()

        # # build adjust layer
        # if cfg.ADJUST.ADJUST:
        #     self.neck = get_neck(cfg.ADJUST.TYPE,
        #                          **cfg.ADJUST.KWARGS)

        # build segment Unet head
        self.seg_head = LW_OpticalFLow()
        self._flow_init_key = None

        self.flow_normalize_seg = transforms.Normalize(
            mean = [-0.0631835013628006, 0.006084870081394911], 
            std = [9.645421028137207, 10.885346412658691])

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    # def forward(self, data):
    #     """ only used in training
    #     """
    #     template = data['template'].cuda()
    #     search = data['search'].cuda()
    #     label_cls = data['label_cls'].cuda()
    #     label_loc = data['label_loc'].cuda()
    #     label_loc_weight = data['label_loc_weight'].cuda()

    #     # get feature
    #     zf = self.backbone(template)
    #     xf = self.backbone(search)
    #     if cfg.MASK.MASK:
    #         zf = zf[-1]
    #         self.xf_refine = xf[:-1]
    #         xf = xf[-1]
    #     if cfg.ADJUST.ADJUST:
    #         zf = self.neck(zf)
    #         xf = self.neck(xf)
    #     cls, loc = self.rpn_head(zf, xf)

    #     # get loss
    #     cls = self.log_softmax(cls)
    #     cls_loss = select_cross_entropy_loss(cls, label_cls)
    #     loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

    #     outputs = {}
    #     outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
    #         cfg.TRAIN.LOC_WEIGHT * loc_loss
    #     outputs['cls_loss'] = cls_loss
    #     outputs['loc_loss'] = loc_loss

    #     if cfg.MASK.MASK:
    #         # TODO
    #         mask, self.mask_corr_feature = self.mask_head(zf, xf)
    #         mask_loss = None
    #         outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
    #         outputs['mask_loss'] = mask_loss
    #     return outputs

    def forward(self, img0, img1, iters_s16=1, iters_s8=8):
        if img0.shape != img1.shape or img0.ndim != 4 or img0.shape[1] != 3:
            raise ValueError('Expected two RGB frames with matching [B, 3, H, W] shapes.')
        batch_size, _, height, width = img0.shape
        if min(height, width) < 32 or height % 16 or width % 16:
            raise ValueError('Frame height and width must be >= 32 and divisible by 16.')
        amp = img0.is_cuda and torch.is_autocast_enabled()
        init_key = (batch_size, height, width, img0.device, amp)
        if self._flow_init_key != init_key:
            self.backbone_op.init_bhwd(batch_size, height, width, img0.device, amp)
            self._flow_init_key = init_key
        flow_list = self.backbone_op(img0, img1, iters_s16=iters_s16, iters_s8=iters_s8,
                                     detach_flow_between_iters=False)
        # The last refined flow is available in both train and eval mode.
        # Mask loss trains every refinement step through the accumulated flow.
        output_seg = self.seg_head(self.flow_normalize_seg(flow_list[-1]))
        return output_seg
