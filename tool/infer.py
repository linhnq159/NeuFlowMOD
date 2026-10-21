"""Predict a current-frame mask or learned bounding boxes from a current/past pair."""

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
                              resolve_stabilization, resolve_seg_input, resolve_task)
from tool.detection import (add_detection_arguments, resolve_detection_options, decode_detections,
                            detection_records, draw_detections)
from tool.training_log import write_json
from NeuFlow.config import SEG_INPUT_MODES
from tool.segmentation_boxes import (add_box_arguments, validate_box_arguments,
                                     contour_boxes, draw_bounding_boxes)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    add_detection_arguments(parser)
    parser.add_argument('--checkpoint', required=True, help='Full ModelBuilder checkpoint; owns the task')
    parser.add_argument('--current', required=True, help='Current RGB or grayscale image')
    parser.add_argument('--past', required=True, help='Past image, with the same original size')
    parser.add_argument('--output', required=True, help='PNG mask (segmentation) or annotated PNG + JSON (detection)')
    parser.add_argument('--resize', '--img_size', dest='img_size', nargs=2, type=int,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Override checkpoint resize; native-size checkpoints keep original frames')
    parser.add_argument('--iters_s16', type=int)
    parser.add_argument('--iters_s8', type=int)
    parser.add_argument('--seg_input', choices=SEG_INPUT_MODES,
                        help='Optional checkpoint architecture check; defaults to checkpoint (legacy: flow)')
    parser.add_argument('--stabilization', choices=['none', 'homography'], default=None,
                        help='Default: checkpoint setting; older checkpoints use none')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action=BooleanOptionalAction, default=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    add_box_arguments(parser)
    args = parser.parse_args(argv)
    validate_box_arguments(args, parser)
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if Path(args.output).suffix.lower() != '.png':
        parser.error('--output must have a .png extension.')
    return args


def resolve_config(args, checkpoint):
    args.task = resolve_task(getattr(args, 'task', None), checkpoint)
    args.seg_input = resolve_seg_input(args.seg_input, checkpoint)
    saved = checkpoint_args(checkpoint)
    resolve_detection_options(args, saved)
    if args.task == 'detection':
        if getattr(args, 'mask_dir', None):
            raise ValueError('--mask_dir is only available for segmentation.')
        if getattr(args, 'draw_boxes', False) or getattr(args, 'min_contour_area', 0) != 0:
            raise ValueError('Detection draws learned boxes automatically; contour options are segmentation-only.')
    for name, default in [('img_size', [960, 544]), ('iters_s16', 1), ('iters_s8', 8)]:
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))
    if args.img_size is not None and (not isinstance(args.img_size, (list, tuple)) or len(args.img_size) != 2
            or any(type(n) is not int or n < 32 for n in args.img_size)):
        raise ValueError('img_size WIDTH HEIGHT must be integers >= 32, or None for native size.')
    for name in ('iters_s16', 'iters_s8'):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    args.stabilization = resolve_stabilization(args.stabilization, checkpoint)
    return args


def run_inference(args):
    paths = [Path(value).expanduser().resolve() for value in (args.checkpoint, args.current, args.past)]
    output_path = Path(args.output).expanduser().resolve()
    boxes_path = (output_path.with_name(f'{output_path.stem}_boxes.png').resolve()
                  if getattr(args, 'draw_boxes', False) else None)
    if output_path in paths:
        raise ValueError('Output mask must not overwrite either input image or the checkpoint.')
    if boxes_path in paths or boxes_path == output_path:
        raise ValueError('Bounding-box image must not overwrite either input image or the checkpoint.')
    checkpoint_path, current_path, past_path = paths
    checkpoint = load_checkpoint(checkpoint_path)
    resolve_config(args, checkpoint)
    json_path = output_path.with_suffix('.json')
    if args.task == 'detection' and json_path in paths:
        raise ValueError('Detection JSON must not overwrite an input or checkpoint.')
    current, past, alignment, original_size = load_frame_pair(
        current_path, past_path, args.img_size, args.stabilization)
    print(alignment.summary())
    print(f'Task: {args.task}; Input size: {current.shape[-1]}x{current.shape[-2]}; iterations S16/S8: {args.iters_s16}/{args.iters_s8}; '
          f'seg_input: {args.seg_input}')
    device = torch.device(args.device)
    model = ModelBuilder(seg_input=args.seg_input, task=args.task).to(device)
    load_model_weights(model, checkpoint)
    model.eval()
    current, past = current.unsqueeze(0).to(device), past.unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            logits = model(current, past, iters_s16=args.iters_s16, iters_s8=args.iters_s8)
        if args.task == 'detection':
            detections = decode_detections(logits, (current.shape[-1], current.shape[-2]), original_size,
                                           args.conf_threshold, args.nms_iou, args.max_detections)[0].cpu()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(current_path) as frame:
                annotated = draw_detections(frame.convert('RGB'), detections, thickness=args.box_thickness)
            Image.fromarray(annotated).save(output_path, format='PNG')
            write_json(json_path, dict(task=args.task, image_size=list(original_size),
                                      detections=detection_records(detections)))
            print(f'Detections: {len(detections)}; image: {output_path}; boxes: {json_path}')
            return detections, alignment
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model produced non-finite mask logits.')
        logits = F.interpolate(logits.float(), size=(original_size[1], original_size[0]),
                               mode='bilinear', align_corners=False)
        mask = (torch.sigmoid(logits[0, 0]) > args.threshold).cpu().numpy().astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path, format='PNG')
    print(f'Mask saved: {output_path} ({original_size[0]}x{original_size[1]})')
    if boxes_path is not None:
        boxes = contour_boxes(mask, args.min_contour_area)
        with Image.open(current_path) as frame:
            annotated = draw_bounding_boxes(frame.convert('RGB'), boxes, args.box_thickness)
        Image.fromarray(annotated).save(boxes_path, format='PNG')
        print(f'Bounding boxes: {len(boxes)}; image saved: {boxes_path}')
    return mask, alignment


def main(argv=None):
    run_inference(parse_args(argv))


if __name__ == '__main__':
    main()
