"""Joint CPU augmentation of aligned RGB frame tensors and a binary target.

Uses torch's RNG, seeded by the trainer/DataLoader and saved in checkpoints.
No input tensor is modified in place. Geometry is shared by both frames and
the target; a lost connected component cancels only the scale/crop operation.
"""

from copy import deepcopy
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from NeuFlow.config import DEFAULT_TRAIN_AUGMENTATION


AUGMENT_MODES = ('hflip', 'brightness_contrast', 'scale')


def validate_augmentation(settings):
    """Return an independent, complete dictionary containing plain Python types."""
    if not isinstance(settings, dict):
        raise ValueError('train_augmentation must be a dictionary.')
    unknown = set(settings) - set(DEFAULT_TRAIN_AUGMENTATION)
    if unknown:
        raise ValueError(f'Unknown train_augmentation keys: {sorted(unknown)}')
    result = deepcopy(DEFAULT_TRAIN_AUGMENTATION)
    result.update(deepcopy(settings))
    enabled = result['enabled']
    if (not isinstance(enabled, (list, tuple))
            or any(value not in AUGMENT_MODES for value in enabled)
            or len(set(enabled)) != len(enabled)):
        raise ValueError(f'augmentation enabled must contain unique choices from {AUGMENT_MODES}.')
    result['enabled'] = list(enabled)
    for key in ('hflip_p', 'brightness_contrast_p', 'zoom_out_p', 'zoom_in_p'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be a finite probability in [0, 1].')
        result[key] = float(value)
    if result['zoom_out_p'] + result['zoom_in_p'] > 1:
        raise ValueError('zoom_out_p + zoom_in_p must be <= 1.')
    if result['zoom_in_p'] > 0 and result['zoom_in_p'] >= result['zoom_out_p']:
        raise ValueError('zoom_in_p must be smaller than zoom_out_p when zoom-in is enabled.')
    for key in ('brightness_range', 'contrast_range', 'zoom_out_range', 'zoom_in_range'):
        value = result[key]
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in value)
                or not 0 < value[0] <= value[1]):
            raise ValueError(f'{key} must be two ordered, finite positive numbers.')
        if key == 'zoom_out_range' and value[1] > 1:
            raise ValueError('zoom_out_range must be <= 1.')
        if key == 'zoom_in_range' and value[0] < 1:
            raise ValueError('zoom_in_range must be >= 1.')
        result[key] = [float(v) for v in value]
    return result


def _uniform(bounds):
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(()).item()


def _offset(limit):
    return int(torch.randint(limit + 1, ()).item())


class FrameMaskAugmentor:
    def __init__(self, settings):
        self.settings = validate_augmentation(settings)

    def _scale(self, current, past, mask, factor):
        height, width = mask.shape[-2:]
        size = (max(1, round(height * factor)), max(1, round(width * factor)))
        if size == (height, width):
            return current, past, mask
        count, labels = cv2.connectedComponents(
            np.ascontiguousarray(mask[0].numpy() > 0, dtype=np.uint8), connectivity=8)
        # Transform IDs, then derive the binary target from them: checking and
        # training must use exactly the same nearest-neighbor sampling.
        ids = F.interpolate(torch.from_numpy(labels).to(torch.float64)[None, None],
                            size=size, mode='nearest-exact')[0, 0].to(torch.int64)
        top, left = _offset(abs(height - size[0])), _offset(abs(width - size[1]))
        if factor > 1:
            ids = ids[top:top + height, left:left + width]
        # Linear counting avoids sorting a full-resolution ID map for each pair.
        if (torch.bincount(ids.reshape(-1), minlength=count)[1:] == 0).any():
            return current, past, mask

        frames = F.interpolate(torch.stack((current, past)), size=size,
                               mode='bilinear', align_corners=False)
        if factor < 1:
            # One shared border color avoids introducing different padding in
            # the two frames, and unlike reflection never duplicates objects.
            color = (current.mean((1, 2)) + past.mean((1, 2))) / 2
            output = color[None, :, None, None].expand(2, 3, height, width).clone()
            output[:, :, top:top + size[0], left:left + size[1]] = frames
            target = torch.zeros_like(mask)
            target[0, top:top + size[0], left:left + size[1]] = ids > 0
        else:
            output = frames[:, :, top:top + height, left:left + width]
            target = (ids > 0).to(mask.dtype).unsqueeze(0)
        return output[0], output[1], target

    def __call__(self, current, past, mask):
        cfg = self.settings
        enabled = cfg['enabled']
        if 'scale' in enabled:
            choice = torch.rand(()).item()
            if choice < cfg['zoom_out_p']:
                current, past, mask = self._scale(current, past, mask, _uniform(cfg['zoom_out_range']))
            elif choice < cfg['zoom_out_p'] + cfg['zoom_in_p']:
                current, past, mask = self._scale(current, past, mask, _uniform(cfg['zoom_in_range']))
        if 'hflip' in enabled and torch.rand(()).item() < cfg['hflip_p']:
            current, past, mask = (value.flip(-1) for value in (current, past, mask))
        if 'brightness_contrast' in enabled and torch.rand(()).item() < cfg['brightness_contrast_p']:
            brightness = _uniform(cfg['brightness_range'])
            contrast = _uniform(cfg['contrast_range'])
            current, past = (((value - 127.5) * contrast + 127.5) * brightness
                             for value in (current, past))
            current, past = current.clamp(0, 255), past.clamp(0, 255)
        return current, past, mask
