"""Current/past frame pairs with single-class YOLO annotations in current coordinates."""

from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_utils.frame_mask_dataset import IMAGE_EXTENSIONS, natural_key
from data_utils.frame_pair import prepare_frame_pair, STABILIZATION_MODES
from data_utils.frame_augmentation import validate_augmentation, _uniform, _offset


def read_yolo(path):
    """Return normalized xyxy boxes. Missing labels are never interpreted as background."""
    boxes = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = [float(value) for value in line.split()]
            if len(fields) != 5 or not np.isfinite(fields).all() or fields[0] != 0:
                raise ValueError('expected class 0 and four finite coordinates')
            _, cx, cy, width, height = fields
            box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
            if width <= 0 or height <= 0 or min(box) < -1e-6 or max(box) > 1 + 1e-6:
                raise ValueError('bbox must have positive size and lie inside [0,1]')
            boxes.append([min(1., max(0., value)) for value in box])
        except ValueError as error:
            raise ValueError(f'Invalid YOLO label {path}:{line_number}: {error}') from error
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def collate_frame_boxes(batch):
    shapes = {tuple(image.shape) for sample in batch for image in sample[:2]}
    if len(shapes) != 1:
        raise ValueError('Frame sizes differ within a batch. Use --resize WIDTH HEIGHT.')
    return torch.stack([s[0] for s in batch]), torch.stack([s[1] for s in batch]), [s[2] for s in batch]


class FrameBoxAugmentor:
    def __init__(self, settings):
        self.settings = validate_augmentation(settings)

    def _scale(self, current, past, boxes, factor):
        height, width = current.shape[-2:]
        h, w = max(1, round(height * factor)), max(1, round(width * factor))
        if (h, w) == (height, width):
            return current, past, boxes
        top, left = _offset(abs(height - h)), _offset(abs(width - w))
        result = boxes * boxes.new_tensor([w / width, h / height, w / width, h / height])
        shift = boxes.new_tensor([left, top, left, top])
        result = result + shift if factor < 1 else result - shift
        result[:, 0::2].clamp_(0, width)
        result[:, 1::2].clamp_(0, height)
        if ((result[:, 2:] - result[:, :2]) <= 0).any():
            return current, past, boxes
        frames = F.interpolate(torch.stack((current, past)), size=(h, w), mode='bilinear', align_corners=False)
        if factor < 1:
            color = (current.mean((1, 2)) + past.mean((1, 2))) / 2
            output = color[None, :, None, None].expand(2, 3, height, width).clone()
            output[:, :, top:top + h, left:left + w] = frames
        else:
            output = frames[:, :, top:top + height, left:left + width]
        return output[0], output[1], result

    def __call__(self, current, past, boxes):
        cfg = self.settings
        if 'scale' in cfg['enabled']:
            choice = torch.rand(()).item()
            if choice < cfg['zoom_out_p']:
                current, past, boxes = self._scale(current, past, boxes, _uniform(cfg['zoom_out_range']))
            elif choice < cfg['zoom_out_p'] + cfg['zoom_in_p']:
                current, past, boxes = self._scale(current, past, boxes, _uniform(cfg['zoom_in_range']))
        if 'hflip' in cfg['enabled'] and torch.rand(()).item() < cfg['hflip_p']:
            current, past = current.flip(-1), past.flip(-1)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = current.shape[-1] - boxes[:, [2, 0]]
        if 'brightness_contrast' in cfg['enabled'] and torch.rand(()).item() < cfg['brightness_contrast_p']:
            brightness, contrast = _uniform(cfg['brightness_range']), _uniform(cfg['contrast_range'])
            current, past = [(((value - 127.5) * contrast + 127.5) * brightness).clamp(0, 255)
                             for value in (current, past)]
        return current, past, boxes


class FrameBoxDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=None, frame_skip=1,
                 split='all', val_split=0.2, stabilization='none', augmentation=None):
        self.image_dir, self.label_dir = Path(image_dir).expanduser().resolve(), Path(label_dir).expanduser().resolve()
        self.img_size = tuple(img_size) if img_size is not None else None
        self.stabilization = stabilization
        self.augmentor = FrameBoxAugmentor(augmentation) if augmentation is not None else None
        if frame_skip < 1 or split not in ('all', 'train', 'val') or not 0 <= val_split < 1:
            raise ValueError('Invalid frame_skip/split/val_split.')
        if stabilization not in STABILIZATION_MODES:
            raise ValueError(f'Unsupported stabilization: {stabilization}')
        if self.img_size is not None and (len(self.img_size) != 2 or any(type(n) is not int or n < 32 for n in self.img_size)):
            raise ValueError('img_size must be None or integer WIDTH HEIGHT >= 32.')
        for path in (self.image_dir, self.label_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if (self.image_dir == self.label_dir or self.image_dir in self.label_dir.parents
                or self.label_dir in self.image_dir.parents):
            raise ValueError('Image and label folders must be separate, non-nested folders.')
        sequences, keys = defaultdict(list), set()
        for path in self.image_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                relative = path.relative_to(self.image_dir)
                key = relative.with_suffix('')
                if key in keys:
                    raise ValueError(f'Ambiguous image stem: {key}')
                keys.add(key)
                sequences[relative.parent].append(path)
        self.samples, self.annotations = [], []
        for sequence in sorted(sequences, key=natural_key):
            frames = sorted(sequences[sequence], key=natural_key)
            if split != 'all' and val_split > 0:
                boundary = int(len(frames) * (1 - val_split))
                frames = frames[:boundary] if split == 'train' else frames[boundary:]
            elif split == 'val':
                frames = []
            for index in range(frame_skip, len(frames)):
                current, past = frames[index], frames[index - frame_skip]
                label = self.label_dir / current.relative_to(self.image_dir).with_suffix('.txt')
                if label.is_file():
                    self.samples.append((current, past, label))
                    self.annotations.append(read_yolo(label))
        if not self.samples:
            raise ValueError('No valid frame pairs with current-frame YOLO TXT labels; check paths/frame_skip/split.')
        self._warned = set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        current_path, past_path, _ = self.samples[index]
        with Image.open(current_path) as current, Image.open(past_path) as past:
            current, past, alignment = prepare_frame_pair(current, past, self.img_size, self.stabilization)
        if self.stabilization == 'homography' and not alignment.success and alignment.reason not in self._warned:
            self._warned.add(alignment.reason)
            warnings.warn(f'{alignment.summary()}; Current={current_path}, past={past_path}', stacklevel=2)
        h, w = current.shape[-2:]
        boxes = self.annotations[index] * torch.tensor([w, h, w, h])
        return self.augmentor(current, past, boxes) if self.augmentor else (current, past, boxes)
