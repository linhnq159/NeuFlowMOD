"""Pairs of video frames with a binary mask in the source frame's coordinates."""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from data_utils.frame_pair import STABILIZATION_MODES, prepare_frame_pair


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
MASK_EXTENSIONS = {'.png', '.bmp'}
MASK_FORMATS = {'PNG', 'BMP'}


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', str(path)))


class FrameMaskDataset(Dataset):
    """Each image subfolder is a sequence; masks match relative path and stem.

    ``frame_skip=1`` pairs adjacent files after natural sorting. Inputs are
    (current frame, past frame), and the mask belongs to the current frame.
    Temporal train/validation splits happen before constructing frame pairs.
    Only current frames with masks become targets; unmasked frames remain
    available as past frames and count toward ``frame_skip``.
    """

    def __init__(self, image_dir, mask_dir, img_size=(960, 544), frame_skip=1,
                 split='all', val_split=0.2, stabilization='none'):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve()
        self.img_size = tuple(img_size)
        if stabilization not in STABILIZATION_MODES:
            raise ValueError(f'Unsupported stabilization: {stabilization}')
        self.stabilization = stabilization
        self._warned_alignment_failures = set()
        if frame_skip < 1:
            raise ValueError('frame_skip must be at least 1 (1 means adjacent frames).')
        if split not in ('all', 'train', 'val') or not 0 <= val_split < 1:
            raise ValueError('Invalid split or val_split; require 0 <= val_split < 1.')
        if len(self.img_size) != 2 or any(size <= 0 for size in self.img_size):
            raise ValueError('img_size must contain positive WIDTH HEIGHT values.')
        for folder in (self.image_dir, self.mask_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f'Data folder does not exist: {folder}')
        if (self.image_dir == self.mask_dir or self.image_dir in self.mask_dir.parents
                or self.mask_dir in self.image_dir.parents):
            raise ValueError('Image and mask folders must be separate, non-nested folders.')

        sequences = defaultdict(list)
        frame_keys = set()
        for path in self.image_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                relative = path.relative_to(self.image_dir)
                key = relative.with_suffix('')
                if key in frame_keys:
                    raise ValueError(f'Ambiguous image stem: {key}')
                frame_keys.add(key)
                sequences[relative.parent].append(path)

        masks = {}
        mask_image_extensions = set(Image.registered_extensions())
        for path in self.mask_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in mask_image_extensions:
                if path.suffix.lower() not in MASK_EXTENSIONS:
                    raise ValueError(f'Masks must use PNG/BMP to preserve labels: {path}')
                key = path.relative_to(self.mask_dir).with_suffix('')
                if key in masks:
                    raise ValueError(f'Ambiguous mask stem: {key}')
                masks[key] = path

        self.samples = []
        for sequence in sorted(sequences, key=natural_key):
            frames = sorted(sequences[sequence], key=natural_key)
            if split != 'all' and val_split > 0:
                boundary = int(len(frames) * (1 - val_split))
                frames = frames[:boundary] if split == 'train' else frames[boundary:]
            elif split == 'val':
                frames = []
            for index in range(frame_skip, len(frames)):
                source, reference = frames[index], frames[index - frame_skip]
                key = source.relative_to(self.image_dir).with_suffix('')
                if key not in masks:
                    continue
                self.samples.append((source, reference, masks[key]))
        if not self.samples:
            raise ValueError(
                f'No frame pairs for split={split!r} in {self.image_dir}. '
                f'Each selected sequence needs more than {frame_skip} frames and '
                'at least one current frame with a matching mask after pairing. '
                'Check mask paths, reduce frame_skip/val_split or provide separate validation folders.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source_path, reference_path, mask_path = self.samples[index]
        with Image.open(source_path) as source_file, Image.open(reference_path) as reference_file, Image.open(mask_path) as mask_file:
            if mask_file.format not in MASK_FORMATS:
                raise ValueError(f'Masks must use PNG/BMP; decoded format is {mask_file.format}: {mask_path}')
            if source_file.size != reference_file.size or source_file.size != mask_file.size:
                raise ValueError(f'Frame/mask sizes differ: {source_path}, {reference_path}, {mask_path}')
            source, reference, alignment = prepare_frame_pair(
                source_file, reference_file, self.img_size, self.stabilization)
            worker = get_worker_info()
            warning_key = (worker.id if worker is not None else None, alignment.reason)
            if (self.stabilization == 'homography' and not alignment.success
                    and warning_key not in self._warned_alignment_failures):
                self._warned_alignment_failures.add(warning_key)
                warnings.warn(f'{alignment.summary()}; using unwarped past. '
                              f'Current={source_path}, past={reference_path}', stacklevel=2)
            # Preserve class IDs, including masks encoded with a PNG palette.
            mask = np.array(mask_file)
            if mask.ndim == 3:
                mask = np.any(mask[..., :3] > 0, axis=-1)
            else:
                mask = mask > 0
            mask = Image.fromarray(mask.astype(np.uint8)).resize(self.img_size, Image.Resampling.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        return source, reference, mask
