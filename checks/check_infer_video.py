"""Video/sequence inference, pairing, precision, masks, reports and resource cleanup."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tool.infer import parse_args as pair_args, run_inference as infer_pair
from tool.infer_video import parse_args, resolve_video_config, run_inference, overlay_frame
from tool.segmentation_boxes import contour_boxes, draw_bounding_boxes


class RecordingModel(nn.Module):
    instances = []

    def __init__(self, seg_input='flow'):
        super().__init__()
        self.seg_input = seg_input
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []
        self.instances.append(self)

    def forward(self, current, past, iters_s16=1, iters_s8=8):
        self.calls.append((current.clone(), past.clone(), iters_s16, iters_s8))
        return ((current[:, :1] - 100) / 30 + (past[:, :1] - 100) / 100) * self.weight


class BoundingBoxChecks(unittest.TestCase):
    def test_disjoint_regions_holes_single_pixels_and_area_filter(self):
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[0, 15] = 255
        mask[2:6, 3:8] = 255
        mask[7:12, 9:16] = 255
        mask[9:11, 11:13] = 0  # a hole must not create a second box
        before = mask.copy()
        self.assertEqual(contour_boxes(mask), [[15, 0, 1, 1], [3, 2, 5, 4], [9, 7, 7, 5]])
        self.assertEqual(contour_boxes(mask, 13), [[9, 7, 7, 5]])
        np.testing.assert_array_equal(mask, before)
        self.assertEqual(contour_boxes((mask > 0).astype(np.uint8)), contour_boxes(mask))

    def test_empty_and_full_frame_masks(self):
        self.assertEqual(contour_boxes(np.zeros((12, 16), dtype=np.uint8)), [])
        self.assertEqual(contour_boxes(np.ones((12, 16), dtype=np.uint8)), [[0, 0, 16, 12]])

    def test_red_box_corners_without_changing_source_or_dimensions(self):
        rgb = np.full((12, 16, 3), 40, dtype=np.uint8)
        before = rgb.copy()
        annotated = draw_bounding_boxes(rgb, [[3, 2, 5, 4]], thickness=1)
        for x, y in ((3, 2), (7, 2), (3, 5), (7, 5)):
            np.testing.assert_array_equal(annotated[y, x], [255, 0, 0])
        np.testing.assert_array_equal(annotated[2, 8], [40, 40, 40])
        np.testing.assert_array_equal(annotated[3, 4], [40, 40, 40])
        np.testing.assert_array_equal(rgb, before)
        np.testing.assert_array_equal(draw_bounding_boxes(rgb, []), rgb)


class VideoChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.checkpoint = self.root / 'model.pth'
        self.saved_args = dict(seg_input='flow_rgb', img_size=[32, 32], frame_skip=2,
                               iters_s16=2, iters_s8=3, stabilization='none', amp=False)
        self.save_checkpoint()
        self.model_patch = patch('tool.infer_video.ModelBuilder', RecordingModel)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        RecordingModel.instances.clear()

    def save_checkpoint(self):
        torch.save(dict(model={'weight': torch.ones(())}, args=self.saved_args, epoch=16), self.checkpoint)

    def make_images(self, size=(48, 32)):
        # Lexicographic order would pair frame_10 before frame_2.
        for i, number in enumerate((1, 2, 10, 11, 20)):
            rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            rgb[:, :, 0] = np.linspace(10 + i * 5, 230, size[0]).astype(np.uint8)
            rgb[:, :, 1] = 40 + i
            rgb[:, :, 2] = 80 + i
            Image.fromarray(rgb).save(self.images / f'frame_{number}.png')

    def args(self, *extra):
        return parse_args(['--checkpoint', str(self.checkpoint), '--input', str(self.images),
                           '--output', str(self.root / 'out.avi'), '--mask_dir', str(self.root / 'masks'),
                           '--device', 'cpu', *extra])

    def run_video(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_inference(self.args(*extra))

    def read_video(self, path):
        capture = cv2.VideoCapture(str(path))
        try:
            self.assertTrue(capture.isOpened())
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            return frames, capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()

    def test_sequence_pairing_rgb_config_masks_and_reports(self):
        self.make_images()
        report = self.run_video('--fps', '12')
        self.assertEqual(report['status'], 'completed')
        self.assertEqual((report['frames_written'], report['initial_frames'], report['predicted_frames']), (5, 2, 3))
        self.assertEqual(report['precision'], 'FP32')
        self.assertEqual(report['alignment_counts'], {'disabled:disabled': 3})
        model = RecordingModel.instances[-1]
        self.assertEqual(model.seg_input, 'flow_rgb')
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.calls[0][2:], (2, 3))
        self.assertEqual(model.calls[0][0].shape, (1, 3, 32, 32))
        self.assertEqual(model.calls[0][0][0, 1, 0, 0], 42)  # current=frame_10 (RGB)
        self.assertEqual(model.calls[0][1][0, 1, 0, 0], 40)  # past=frame_1
        masks = sorted((self.root / 'masks').glob('*.png'))
        self.assertEqual([p.name for p in masks], ['frame_000002.png', 'frame_000003.png', 'frame_000004.png'])
        for p in masks:
            with Image.open(p) as mask:
                self.assertEqual(mask.size, (48, 32))
                self.assertEqual(set(np.unique(mask)), {0, 255})
        frames, fps = self.read_video(self.root / 'out.avi')
        self.assertEqual(len(frames), 5)
        self.assertEqual(fps, 12)
        self.assertEqual(frames[0].shape, (32, 48, 3))
        with (self.root / 'out.timings.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r['source'] for r in rows], ['frame_1.png', 'frame_2.png', 'frame_10.png', 'frame_11.png', 'frame_20.png'])
        self.assertEqual(rows[2]['past_source'], 'frame_1.png')
        self.assertEqual(rows[0]['foreground_pixels'], '')
        self.assertEqual(json.loads((self.root / 'out.stats.json').read_text()), report)
        self.assertGreater(report['mean_forward_ms'], 0)
        self.assertGreater(report['pipeline_fps'], 0)
        self.assertGreaterEqual(report['wall_seconds'], report['processing_seconds'])

    def test_pair_inference_parity_after_resize_and_threshold(self):
        self.make_images(size=(47, 35))
        self.run_video('--threshold', '0.35')
        with patch('tool.infer.ModelBuilder', RecordingModel), contextlib.redirect_stdout(io.StringIO()):
            expected, _ = infer_pair(pair_args([
                '--checkpoint', str(self.checkpoint), '--current', str(self.images / 'frame_10.png'),
                '--past', str(self.images / 'frame_1.png'), '--output', str(self.root / 'pair.png'),
                '--device', 'cpu', '--no-amp', '--threshold', '0.35']))
        with Image.open(self.root / 'masks/frame_000002.png') as image:
            np.testing.assert_array_equal(np.array(image), expected)
            self.assertEqual(image.size, (47, 35))
        frames, _ = self.read_video(self.root / 'out.avi')
        self.assertEqual(frames[0].shape, (36, 48, 3))  # encoder padding, not a crop

    def test_pair_bounding_box_sidecar_uses_original_size_and_keeps_binary_mask(self):
        self.make_images(size=(96, 64))
        command = ['--checkpoint', str(self.checkpoint), '--current', str(self.images / 'frame_10.png'),
                   '--past', str(self.images / 'frame_1.png'), '--output', str(self.root / 'pair.png'),
                   '--device', 'cpu', '--no-amp']
        with patch('tool.infer.ModelBuilder', RecordingModel), contextlib.redirect_stdout(io.StringIO()):
            baseline, _ = infer_pair(pair_args(command))
            self.assertFalse((self.root / 'pair_boxes.png').exists())
            mask, _ = infer_pair(pair_args(command + ['--draw_boxes', '--box_thickness', '1']))
        np.testing.assert_array_equal(mask, baseline)
        boxes = contour_boxes(mask)
        self.assertEqual(len(boxes), 1)
        x, y, width, height = boxes[0]
        self.assertEqual((y, height, x + width), (0, 64, 96))
        with Image.open(self.root / 'pair_boxes.png') as annotated:
            self.assertEqual(annotated.size, (96, 64))
            self.assertEqual(annotated.mode, 'RGB')
            self.assertEqual(annotated.getpixel((x, y)), (255, 0, 0))
        with Image.open(self.root / 'pair.png') as saved:
            self.assertEqual(saved.mode, 'L')
            np.testing.assert_array_equal(np.array(saved), mask)

    def test_video_boxes_counts_coordinates_and_masks(self):
        self.make_images()
        def logits(current, past, **kwargs):
            result = torch.full_like(current[:, :1], -10.)
            result[:, :, 4:10, 6:14] = 10
            result[:, :, 20, 30] = 10
            return result
        with patch.object(RecordingModel, 'forward', side_effect=logits), \
                patch('tool.infer_video.draw_bounding_boxes', wraps=draw_bounding_boxes) as draw:
            report = self.run_video('--draw_boxes', '--resize', '48', '32', '--box_thickness', '1')
        self.assertEqual(draw.call_count, 3)
        self.assertEqual(draw.call_args.args[1], [[6, 4, 8, 6], [30, 20, 1, 1]])
        self.assertEqual(report['boxes_drawn'], 6)
        with (self.root / 'out.timings.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['box_count'], '')
        self.assertEqual(rows[2]['box_count'], '2')
        self.assertEqual(json.loads(rows[2]['boxes_xywh']), [[6, 4, 8, 6], [30, 20, 1, 1]])
        with Image.open(self.root / 'masks/frame_000002.png') as saved:
            mask = np.array(saved)
        self.assertEqual(np.count_nonzero(mask), 49)
        self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_box_cli_validation_and_pair_output_protection(self):
        self.make_images()
        command = ['--checkpoint', str(self.checkpoint), '--current', str(self.images / 'frame_10.png'),
                   '--past', str(self.images / 'frame_1.png'), '--output', str(self.root / 'pair.png')]
        for parser, base in ((pair_args, command), (self.args, [])):
            for flags in (('--min_contour_area', '-1'), ('--min_contour_area', 'nan'),
                          ('--min_contour_area', 'inf'), ('--box_thickness', '0')):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser([*base, *flags]) if parser is pair_args else parser(*flags)
        # A source named like the derived sidecar must be protected before any save.
        current = self.root / 'pair_boxes.png'
        current.write_bytes((self.images / 'frame_10.png').read_bytes())
        before = current.read_bytes()
        args = pair_args([*command, '--current', str(current), '--draw_boxes'])
        with self.assertRaisesRegex(ValueError, 'Bounding-box image'):
            infer_pair(args)
        self.assertEqual(current.read_bytes(), before)
        self.assertFalse((self.root / 'pair.png').exists())

    def test_real_video_input_preserves_frame_rate_and_frame_limit(self):
        path = self.root / 'input.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 17., (48, 32))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(7):
                writer.write(np.full((32, 48, 3), 30 + i * 20, dtype=np.uint8))
        finally:
            writer.release()
        report = self.run_video('--input', str(path), '--max_frames', '5', '--frame_skip', '3')
        self.assertEqual(report['kind'], 'video')
        self.assertEqual((report['frames_written'], report['predicted_frames']), (5, 2))
        frames, fps = self.read_video(self.root / 'out.avi')
        self.assertEqual((len(frames), fps), (5, 17))
        first = RecordingModel.instances[-1].calls[0]
        self.assertAlmostEqual(first[0].mean().item(), 90, delta=3)
        self.assertAlmostEqual(first[1].mean().item(), 30, delta=3)

    def test_native_size_checkpoint_and_cpu_amp_override(self):
        self.make_images(size=(47, 35))
        self.saved_args.update(img_size=None, amp=True)
        self.save_checkpoint()
        report = self.run_video('--frame_skip', '1', '--max_frames', '2')
        self.assertEqual(report['precision'], 'FP32')
        self.assertEqual(RecordingModel.instances[-1].calls[0][0].shape, (1, 3, 35, 47))

    def test_stabilization_fallback_and_mp4_output(self):
        self.make_images()
        report = self.run_video('--output', str(self.root / 'out.mp4'), '--stabilization', 'homography',
                                '--max_frames', '3')
        self.assertEqual(report['alignment_counts'], {'fallback:insufficient_features': 1})
        frames, _ = self.read_video(self.root / 'out.mp4')
        self.assertEqual(len(frames), 3)

    def test_real_model_native_non_multiple_size(self):
        from NeuFlow.model_builder import ModelBuilder

        self.make_images(size=(47, 35))
        torch.manual_seed(1)
        model = ModelBuilder(seg_input='flow_rgb')
        self.saved_args.update(img_size=None, iters_s16=1, iters_s8=1)
        torch.save(dict(model=model.state_dict(), args=self.saved_args), self.checkpoint)
        with patch('tool.infer_video.ModelBuilder', ModelBuilder):
            report = self.run_video('--max_frames', '3')
        self.assertEqual(report['predicted_frames'], 1)
        self.assertEqual(report['original_size'], [47, 35])
        with Image.open(self.root / 'masks/frame_000002.png') as image:
            self.assertEqual(image.size, (47, 35))
            self.assertTrue(set(np.unique(image)) <= {0, 255})

    def test_keyboard_interrupt_marks_partial_output(self):
        self.make_images()
        with patch.object(RecordingModel, 'forward', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_video()
        report = json.loads((self.root / 'out.stats.json').read_text())
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['frames_written'], 2)

    def test_nonfinite_logits_report_failed_frame_and_release_video(self):
        self.make_images()
        with patch.object(RecordingModel, 'forward', side_effect=lambda a, b, **kw: torch.full_like(a[:, :1], float('nan'))):
            with self.assertRaisesRegex(RuntimeError, 'current=frame_10.png, past=frame_1.png.*non-finite'):
                self.run_video()
        report = json.loads((self.root / 'out.stats.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['frames_written'], 2)
        self.assertIsNone(report['mean_forward_ms'])
        frames, _ = self.read_video(self.root / 'out.avi')
        self.assertEqual(len(frames), 2)

    def test_insufficient_or_mixed_size_inputs(self):
        self.make_images()
        with self.assertRaisesRegex(ValueError, 'more than frame_skip'):
            self.run_video('--frame_skip', '5')
        self.assertFalse((self.root / 'out.avi').exists())
        Image.new('RGB', (50, 34)).save(self.images / 'frame_10.png')
        with self.assertRaisesRegex(ValueError, 'Frame size changed'):
            self.run_video()
        self.assertEqual(json.loads((self.root / 'out.stats.json').read_text())['status'], 'failed')

    def test_invalid_cli_and_checkpoint_settings(self):
        for flags in (('--threshold', 'nan'), ('--fps', '0'), ('--fps', 'nan'),
                      ('--frame_skip', '0'), ('--max_frames', '0'), ('--codec', 'x'),
                      ('--output', 'x.png'), ('--overlay_alpha', '2')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*flags)
        for flags in (('--max_frames', '2'), ('--seg_input', 'flow'), ('--resize', '1', '2')):
            with self.assertRaises(ValueError):
                resolve_video_config(self.args(*flags), {'args': self.saved_args})

    def test_legacy_defaults_precision_and_overrides(self):
        args = resolve_video_config(self.args(), {})
        self.assertEqual((args.frame_skip, args.img_size, args.amp), (1, [960, 544], False))
        args = resolve_video_config(self.args('--amp', '--frame_skip', '4', '--iters_s8', '10'),
                                    {'args': self.saved_args})
        self.assertEqual((args.amp, args.frame_skip, args.iters_s8), (True, 4, 10))

    def test_output_protection(self):
        self.make_images()
        with self.assertRaisesRegex(ValueError, 'mask_dir'):
            self.run_video('--mask_dir', str(self.images))
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            self.run_video('--input', str(self.root / 'out.avi'))
        masks = self.root / 'masks'
        masks.mkdir()
        (masks / 'previous.png').write_bytes(b'previous run')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            self.run_video()
        self.assertEqual((masks / 'previous.png').read_bytes(), b'previous run')

    def test_overlay_changes_only_foreground(self):
        frame = Image.fromarray(np.full((4, 5, 3), 100, dtype=np.uint8))
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1, 2] = 255
        result = overlay_frame(frame, mask, .5)
        np.testing.assert_array_equal(result[1, 2], [50, 178, 50])
        np.testing.assert_array_equal(result[mask == 0], np.full((19, 3), 100))
        np.testing.assert_array_equal(overlay_frame(frame, None, .5), np.asarray(frame))


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    unittest.main(verbosity=2)
