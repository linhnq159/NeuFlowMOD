"""Flow geometry shared by the NeuFlow modules."""

import torch
import torch.nn.functional as F


def coords_grid(b, h, w, device, amp):
    dtype = torch.half if amp else torch.float
    ys, xs = torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                            torch.arange(w, dtype=dtype, device=device), indexing='ij')
    return torch.stack([xs, ys], dim=0)[None].repeat(b, 1, 1, 1)


def bilinear_sample(img, sample_coords):
    _, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def flow_warp(feature, flow):
    b, _, h, w = feature.shape
    grid = coords_grid(b, h, w, flow.device, flow.dtype == torch.float16) + flow
    return bilinear_sample(feature, grid)
