"""Predict the current-frame segmentation mask from a current/past image pair."""

import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data_utils.frame_pair import load_frame_pair
from NeuFlow.model_builder import ModelBuilder
from tool.checkpoints import (checkpoint_args, load_checkpoint, load_model_weights,
                              resolve_stabilization, resolve_seg_input)
from NeuFlow.config import SEG_INPUT_MODES


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Full ModelBuilder checkpoint')
    parser.add_argument('--current', required=True, help='Current RGB or grayscale image')
    parser.add_argument('--past', required=True, help='Past image, with the same original size')
    parser.add_argument('--output', required=True, help='Output PNG mask in current-frame coordinates')
    parser.add_argument('--img_size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--iters_s16', type=int)
    parser.add_argument('--iters_s8', type=int)
    parser.add_argument('--seg_input', choices=SEG_INPUT_MODES,
                        help='Optional checkpoint architecture check; defaults to checkpoint (legacy: flow)')
    parser.add_argument('--stabilization', choices=['none', 'homography'], default=None,
                        help='Default: checkpoint setting; older checkpoints use none')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action=BooleanOptionalAction, default=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if Path(args.output).suffix.lower() != '.png':
        parser.error('--output must have a .png extension.')
    return args


def resolve_config(args, checkpoint):
    args.seg_input = resolve_seg_input(args.seg_input, checkpoint)
    saved = checkpoint_args(checkpoint)
    for name, default in [('img_size', [960, 544]), ('iters_s16', 1), ('iters_s8', 8)]:
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))
    if (not isinstance(args.img_size, (list, tuple)) or len(args.img_size) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 32 or n % 16
                   for n in args.img_size)):
        raise ValueError('img_size WIDTH HEIGHT must be integers >= 32 and divisible by 16.')
    for name in ('iters_s16', 'iters_s8'):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    args.stabilization = resolve_stabilization(args.stabilization, checkpoint)
    return args


def run_inference(args):
    paths = [Path(value).expanduser().resolve() for value in (args.checkpoint, args.current, args.past)]
    output_path = Path(args.output).expanduser().resolve()
    if output_path in paths:
        raise ValueError('Output mask must not overwrite either input image or the checkpoint.')
    checkpoint_path, current_path, past_path = paths
    checkpoint = load_checkpoint(checkpoint_path)
    resolve_config(args, checkpoint)
    current, past, alignment, original_size = load_frame_pair(
        current_path, past_path, args.img_size, args.stabilization)
    print(alignment.summary())
    print(f'Input size: {args.img_size}; iterations S16/S8: {args.iters_s16}/{args.iters_s8}; '
          f'seg_input: {args.seg_input}')
    device = torch.device(args.device)
    model = ModelBuilder(seg_input=args.seg_input).to(device)
    load_model_weights(model, checkpoint)
    model.eval()
    current, past = current.unsqueeze(0).to(device), past.unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            logits = model(current, past, iters_s16=args.iters_s16, iters_s8=args.iters_s8)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model produced non-finite mask logits.')
        logits = F.interpolate(logits.float(), size=(original_size[1], original_size[0]),
                               mode='bilinear', align_corners=False)
        mask = (torch.sigmoid(logits[0, 0]) > args.threshold).cpu().numpy().astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path, format='PNG')
    print(f'Mask saved: {output_path} ({original_size[0]}x{original_size[1]})')
    return mask, alignment


def main(argv=None):
    run_inference(parse_args(argv))


if __name__ == '__main__':
    main()
