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
