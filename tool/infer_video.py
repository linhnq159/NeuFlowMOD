"""Infer segmentation masks or learned boxes on a video or image sequence."""

import csv
from collections import Counter, deque
from contextlib import contextmanager
import json
from itertools import chain, islice
import math
from pathlib import Path
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data_utils.frame_mask_dataset import IMAGE_EXTENSIONS, natural_key
from data_utils.frame_pair import prepare_frame_pair
from NeuFlow.config import SEG_INPUT_MODES
from NeuFlow.model_builder import ModelBuilder
from tool.checkpoints import checkpoint_args, load_checkpoint, load_model_weights
from tool.infer import resolve_config
from tool.detection import add_detection_arguments, decode_detections, draw_detections
from tool.training_log import write_json
from tool.segmentation_boxes import (add_box_arguments, validate_box_arguments,
                                     contour_boxes, draw_bounding_boxes)


TIMING_FIELDS = ('frame_index', 'source', 'past_index', 'past_source', 'predicted',
                 'foreground_pixels', 'box_count', 'boxes_xywh', 'box_scores', 'alignment', 'alignment_reason',
                 'decode_ms', 'preprocess_ms', 'transfer_ms', 'forward_ms',
                 'postprocess_ms', 'write_ms', 'frame_ms')


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    add_detection_arguments(parser)
    parser.add_argument('--checkpoint', required=True, help='Full ModelBuilder checkpoint; owns the task')
    parser.add_argument('--input', required=True, help='Video file or one folder of naturally sorted frames')
    parser.add_argument('--output', required=True, help='Overlay video (.mp4 or .avi); no audio')
    parser.add_argument('--mask_dir', help='Optional directory of original-size binary PNG masks')
    parser.add_argument('--frame_skip', type=int, help='Pair (t, t-frame_skip); default: checkpoint, legacy: 1')
    parser.add_argument('--resize', '--img_size', dest='img_size', nargs=2, type=int,
                        metavar=('WIDTH', 'HEIGHT'), help='Override checkpoint input size')
    parser.add_argument('--iters_s16', type=int)
    parser.add_argument('--iters_s8', type=int)
    parser.add_argument('--seg_input', choices=SEG_INPUT_MODES, help='Optional checkpoint architecture check')
    parser.add_argument('--stabilization', choices=['none', 'homography'], default=None,
                        help='Default: checkpoint setting; use none for already aligned frames')
    parser.add_argument('--amp', action=BooleanOptionalAction, default=None,
                        help='Default: checkpoint AMP setting, or FP32 for legacy checkpoints; CUDA only')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--overlay_alpha', type=float, default=0.45)
    parser.add_argument('--fps', type=float, help='Output playback FPS; default: video FPS or 30 for images')
    parser.add_argument('--max_frames', type=int, help='Process at most N source frames, including initial frames')
    parser.add_argument('--codec', help='Optional four-character codec; default: mp4v for MP4, MJPG for AVI')
    add_box_arguments(parser)
    args = parser.parse_args(argv)
    validate_box_arguments(args, parser)
    for name in ('threshold', 'overlay_alpha'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name} must be between 0 and 1.')
    if args.frame_skip is not None and args.frame_skip < 1:
        parser.error('--frame_skip must be >= 1.')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max_frames must be >= 1.')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and > 0.')
    if args.codec is not None and (len(args.codec) != 4 or not args.codec.isascii()):
        parser.error('--codec must contain exactly four ASCII characters.')
    if Path(args.output).suffix.lower() not in ('.mp4', '.avi'):
        parser.error('--output must have an .mp4 or .avi extension.')
    return args


def resolve_video_config(args, checkpoint):
    resolve_config(args, checkpoint)
    saved = checkpoint_args(checkpoint)
    if args.frame_skip is None:
        args.frame_skip = saved.get('frame_skip', 1)
    if type(args.frame_skip) is not int or args.frame_skip < 1:
        raise ValueError('frame_skip must be an integer >= 1.')
    if args.max_frames is not None and args.max_frames <= args.frame_skip:
        raise ValueError('--max_frames must exceed frame_skip to produce a prediction.')
    if args.amp is None:
        args.amp = saved.get('amp', False)
    if type(args.amp) is not bool:
        raise ValueError('Checkpoint amp must be boolean; override with --amp or --no-amp.')
    return args


@contextmanager
def open_source(path, fps_override=None):
    """Stream RGB PIL frames; never load a whole video into memory."""
    capture = None
    try:
        if path.is_dir():
            files = sorted((p for p in path.iterdir() if p.is_file()
                            and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)
            if not files:
                raise ValueError(f'No images directly inside {path}; select one sequence folder.')

            def frames():
                for index, image_path in enumerate(files):
                    started = time.perf_counter()
                    with Image.open(image_path) as frame:
                        rgb = frame.convert('RGB')
                    yield index, image_path.name, rgb, (time.perf_counter() - started) * 1000

            info = dict(kind='images', source_fps=None, reported_frames=len(files))
            fps = fps_override if fps_override is not None else 30.0
        elif path.is_file():
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f'Cannot open input video: {path}')
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else None
            fps = fps_override if fps_override is not None else source_fps
            if fps is None:
                raise ValueError('Video FPS is unavailable; specify --fps.')
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            info = dict(kind='video', source_fps=source_fps,
                        reported_frames=int(count) if math.isfinite(count) and count > 0 else None)

            def frames():
                index = 0
                while True:
                    started = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    yield index, f'frame_{index:06d}', rgb, (time.perf_counter() - started) * 1000
                    index += 1
        else:
            raise FileNotFoundError(f'Input does not exist: {path}')
        yield iter(frames()), float(fps), info
    finally:
        if capture is not None:
            capture.release()


def overlay_frame(frame, mask, alpha):
    rgb = np.asarray(frame).copy()
    if mask is not None:
        foreground = mask > 0
        rgb[foreground] = np.rint((1 - alpha) * rgb[foreground]
                                  + alpha * np.array([0, 255, 0])).astype(np.uint8)
    return rgb


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_pair(model, current, past, args, device):
    times = {}
    started = time.perf_counter()
    current_tensor, past_tensor, alignment = prepare_frame_pair(
        current, past, args.img_size, args.stabilization)
    times['preprocess_ms'] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    inputs = [v.unsqueeze(0).to(device) for v in (current_tensor, past_tensor)]
    _sync(device)
    times['transfer_ms'] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
        logits = model(*inputs, iters_s16=args.iters_s16, iters_s8=args.iters_s8)
    _sync(device)
    times['forward_ms'] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if args.task == 'detection':
        prediction = decode_detections(logits, (current_tensor.shape[-1], current_tensor.shape[-2]), current.size,
                                        args.conf_threshold, args.nms_iou, args.max_detections)[0].cpu()
        times['postprocess_ms'] = (time.perf_counter() - started) * 1000
        return prediction, alignment, times
    if not torch.isfinite(logits).all():
        raise RuntimeError('Model produced non-finite mask logits.')
    logits = F.interpolate(logits.float(), size=(current.height, current.width),
                           mode='bilinear', align_corners=False)
    mask = (logits[0, 0].sigmoid() > args.threshold).cpu().numpy().astype(np.uint8) * 255
    times['postprocess_ms'] = (time.perf_counter() - started) * 1000
    return mask, alignment, times


def run_inference(args):
    started = time.perf_counter()
    checkpoint_path, input_path, output_path = (
        Path(value).expanduser().resolve() for value in (args.checkpoint, args.input, args.output))
    stats_path = output_path.with_suffix('.stats.json')
    timings_path = output_path.with_suffix('.timings.csv')
    mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None
    protected = {checkpoint_path, input_path}
    if any(path in protected for path in (output_path, stats_path, timings_path)):
        raise ValueError('Outputs must not overwrite the input or checkpoint.')
    if mask_dir is not None and (mask_dir in protected or mask_dir in input_path.parents
                                 or mask_dir in checkpoint_path.parents
                                 or mask_dir in (output_path, stats_path, timings_path)):
        raise ValueError('mask_dir must be separate from the input and checkpoint folders.')
    if mask_dir is not None and mask_dir.exists() and (not mask_dir.is_dir() or any(mask_dir.iterdir())):
        raise ValueError('mask_dir must be a new or empty directory to avoid mixing masks from different runs.')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')
    checkpoint = load_checkpoint(checkpoint_path)
    resolve_video_config(args, checkpoint)
    device = torch.device(args.device)
    model = ModelBuilder(seg_input=args.seg_input, task=args.task).to(device)
    load_model_weights(model, checkpoint)
    model.eval()
    _sync(device)
    setup_seconds = time.perf_counter() - started
    precision = 'AMP FP16' if args.amp and device.type == 'cuda' else 'FP32'
    print(f'Task: {args.task}; Input: {input_path}; resize: {args.img_size or "native"}; frame_skip: {args.frame_skip}; '
          f'seg_input: {args.seg_input}; stabilization: {args.stabilization}; precision: {precision}; '
          f'threshold: {args.conf_threshold if args.task == "detection" else args.threshold}')
    report = dict(status='running', checkpoint=str(checkpoint_path), checkpoint_epoch=checkpoint.get('epoch'),
                  input=str(input_path), output=str(output_path), timings=str(timings_path),
                  config=vars(args).copy(), precision=precision, setup_seconds=setup_seconds,
                  frames_written=0, predicted_frames=0, initial_frames=0,
                  boxes_drawn=0 if args.draw_boxes or args.task == 'detection' else None)
    alignment_counts = Counter()
    totals = Counter()
    writer = None
    processing_started = time.perf_counter()
    # Errors before entering this block leave existing outputs untouched.
    with open_source(input_path, args.fps) as (frames, fps, source_info):
        report.update(source_info, output_fps=fps)
        # Verify enough frames for one prediction before opening output files.
        initial = [next(frames, None) for _ in range(args.frame_skip + 1)]
        if any(item is None for item in initial):
            raise ValueError(f'Input needs more than frame_skip={args.frame_skip} frames.')
        original_size = initial[0][2].size
        encoded_size = tuple(value + value % 2 for value in original_size)
        report.update(original_size=list(original_size), encoded_size=list(encoded_size))
        codec = args.codec or ('mp4v' if output_path.suffix.lower() == '.mp4' else 'MJPG')
        report['codec'] = codec
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
        try:
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, encoded_size)
            if not writer.isOpened():
                raise RuntimeError(f'Cannot create {output_path} with codec {codec}. Try .avi with MJPG.')
            with timings_path.open('w', newline='') as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
                csv_writer.writeheader()
                history = deque(maxlen=args.frame_skip)
                source = chain(initial, frames)
                if args.max_frames is not None:
                    source = islice(source, args.max_frames)
                for index, name, current, decode_ms in source:
                    frame_started = time.perf_counter()
                    if current.size != original_size:
                        raise ValueError(f'Frame size changed at {name}: {current.size}, expected {original_size}.')
                    row = dict.fromkeys(TIMING_FIELDS)
                    row.update(frame_index=index, source=name, predicted=False, decode_ms=decode_ms,
                               preprocess_ms=0., transfer_ms=0., forward_ms=0., postprocess_ms=0.)
                    mask = None
                    if len(history) == args.frame_skip:
                        past_index, past_name, past = history[0]
                        try:
                            mask, alignment, timings = predict_pair(model, current, past, args, device)
                        except Exception as error:
                            raise RuntimeError(f'Inference failed: current={name}, past={past_name}: {error}') from error
                        row.update(timings, predicted=True, past_index=past_index, past_source=past_name,
                                   foreground_pixels=int(np.count_nonzero(mask)) if args.task == 'segmentation' else None,
                                   alignment=('disabled' if alignment.mode == 'none' else
                                              'aligned' if alignment.success else 'fallback'),
                                   alignment_reason=alignment.reason)
                    write_started = time.perf_counter()
                    overlay = overlay_frame(current, mask if args.task == 'segmentation' else None, args.overlay_alpha)
                    if mask is not None and args.task == 'detection':
                        overlay = draw_detections(overlay, mask, thickness=args.box_thickness)
                        boxes = mask[:, :4].clone()
                        boxes[:, 2:] -= boxes[:, :2]
                        row.update(box_count=len(mask), boxes_xywh=json.dumps(boxes.tolist(), separators=(',', ':')),
                                   box_scores=json.dumps(mask[:, 4].tolist(), separators=(',', ':')))
                    elif mask is not None and args.draw_boxes:
                        boxes = contour_boxes(mask, args.min_contour_area)
                        overlay = draw_bounding_boxes(overlay, boxes, args.box_thickness)
                        row.update(box_count=len(boxes), boxes_xywh=json.dumps(boxes, separators=(',', ':')))
                    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    if encoded_size != original_size:
                        bgr = cv2.copyMakeBorder(bgr, 0, encoded_size[1] - original_size[1],
                                                 0, encoded_size[0] - original_size[0], cv2.BORDER_REPLICATE)
                    writer.write(bgr)
                    if mask is not None and mask_dir is not None:
                        Image.fromarray(mask).save(mask_dir / f'frame_{index:06d}.png')
                    row['write_ms'] = (time.perf_counter() - write_started) * 1000
                    row['frame_ms'] = decode_ms + (time.perf_counter() - frame_started) * 1000
                    csv_writer.writerow(row)
                    stream.flush()
                    report['frames_written'] += 1
                    report['predicted_frames' if mask is not None else 'initial_frames'] += 1
                    if row['box_count'] is not None:
                        report['boxes_drawn'] += row['box_count']
                    if mask is not None:
                        alignment_counts[f'{row["alignment"]}:{row["alignment_reason"]}'] += 1
                    for field in TIMING_FIELDS:
                        if field.endswith('_ms'):
                            totals[field] += row[field]
                    history.append((index, name, current))
                    if report['frames_written'] % 100 == 0:
                        print(f'Processed {report["frames_written"]} frames; predicted {report["predicted_frames"]}')
            expected = source_info['reported_frames']
            if args.max_frames is not None and expected is not None:
                expected = min(expected, args.max_frames)
            if expected is not None and report['frames_written'] < expected:
                # OpenCV reports EOF and decode errors identically; expose a
                # metadata mismatch instead of silently declaring a complete video.
                report['status'] = 'incomplete'
                report['warning'] = f'Decoded {report["frames_written"]} of {expected} reported frames.'
            else:
                report['status'] = 'completed'
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise
        finally:
            if writer is not None:
                writer.release()
            seconds = time.perf_counter() - processing_started
            predicted = report['predicted_frames']
            report.update(processing_seconds=seconds, wall_seconds=time.perf_counter() - started,
                          pipeline_fps=report['frames_written'] / seconds if seconds else None,
                          predictions_per_second=predicted / seconds if seconds else None,
                          mean_forward_ms=totals['forward_ms'] / predicted if predicted else None,
                          first_prediction_included=True, stage_totals_ms=dict(totals),
                          alignment_counts=dict(alignment_counts))
            write_json(stats_path, report)
    print(f'{report["status"]}: {report["frames_written"]} frames, {report["predicted_frames"]} predictions; '
          f'pipeline {report["pipeline_fps"]:.2f} FPS; mean forward {report["mean_forward_ms"]:.2f} ms')
    print(f'Overlay: {output_path}\nStatistics: {stats_path}\nPer-frame timings: {timings_path}')
    if report.get('warning'):
        print(report['warning'])
    return report


def main(argv=None):
    run_inference(parse_args(argv))


if __name__ == '__main__':
    main()
