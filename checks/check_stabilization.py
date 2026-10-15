"""Run with: python checks/check_stabilization.py (OpenCV + PyTorch environment)."""

import contextlib
import io
import sys
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils.frame_pair import (_valid_geometry, load_frame_pair, prepare_frame_pair,
                                   stabilize_past, warp_past_to_current)
from data_utils.frame_mask_dataset import FrameMaskDataset
from tool.checkpoints import load_checkpoint, resolve_stabilization
from tool.infer import parse_args as infer_args, resolve_config, run_inference
from tool.train import Trainer, parse_args as train_args


def textured_pair():
    rng = np.random.default_rng(1211)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (384, 512), dtype=np.uint8), (3, 3), .6)
    current = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    current_to_past = np.array([[.99, .025, 10.], [-.018, 1.01, 6.], [.00003, -.00004, 1.]])
    past = cv2.warpPerspective(current, current_to_past, (512, 384))
    return current, past, np.linalg.inv(current_to_past)


class AlignmentChecks(unittest.TestCase):
    def test_past_to_current_homography_and_estimation_resize(self):
        current, past, expected = textured_pair()
        current_before, past_before = current.copy(), past.copy()
        for max_side in (960, 256):
            with self.subTest(max_side=max_side), patch('data_utils.frame_pair.MAX_ESTIMATION_SIDE', max_side):
                aligned, info = stabilize_past(current, past)
                self.assertTrue(info.success, info.summary())
                points = np.float32([[[80, 80], [430, 80], [430, 300], [80, 300]]])
                actual = cv2.perspectiveTransform(points, info.homography)
                desired = cv2.perspectiveTransform(points, expected)
                self.assertLess(float(np.linalg.norm(actual - desired, axis=-1).max()), 1.0)
                before = np.abs(current.astype(float) - past).mean()
                after = np.abs(current.astype(float) - aligned).mean()
                self.assertLess(after, before * .5)
                self.assertGreaterEqual(info.overlap, .5)
        np.testing.assert_array_equal(current, current_before)
        np.testing.assert_array_equal(past, past_before)

    def test_warp_only_past_and_fill_current_at_borders(self):
        rng = np.random.default_rng(1)
        current = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        past = rng.integers(0, 256, current.shape, dtype=np.uint8)
        current_before, past_before = current.copy(), past.copy()
        transform = np.array([[1., 0, 7], [0, 1, 3], [0, 0, 1]])
        aligned, valid = warp_past_to_current(current, past, transform)
        np.testing.assert_array_equal(aligned[~valid], current[~valid])
        np.testing.assert_array_equal(aligned[3:, 7:], past[:-3, :-7])
        np.testing.assert_array_equal(current, current_before)
        np.testing.assert_array_equal(past, past_before)

    def test_flat_or_unmatched_images_fall_back(self):
        flat = np.zeros((64, 96, 3), dtype=np.uint8)
        aligned, info = stabilize_past(flat, flat + 30)
        self.assertFalse(info.success)
        self.assertEqual(info.reason, 'insufficient_features')
        np.testing.assert_array_equal(aligned, flat + 30)
        current, past, _ = textured_pair()
        with patch('data_utils.frame_pair._ratio_matches', return_value={}):
            aligned, info = stabilize_past(current, past)
        self.assertEqual(info.reason, 'insufficient_matches')
        np.testing.assert_array_equal(aligned, past)

    def test_invalid_homography_geometry(self):
        cases = [np.zeros((3, 3)), np.full((3, 3), np.nan),
                 np.array([[-1., 0, 512], [0, 1, 0], [0, 0, 1]]),
                 np.array([[1., 0, 0], [0, 1, 0], [.01, 0, -1.]])]
        for transform in cases:
            with self.subTest(transform=transform.tolist()):
                self.assertFalse(_valid_geometry(transform, 512, 384))
        self.assertTrue(_valid_geometry(np.eye(3), 512, 384))

    def test_rejected_estimates_return_unwarped_past(self):
        current, past, _ = textured_pair()
        cases = [
            ('insufficient_inliers', np.eye(3), False),
            ('invalid_homography', np.zeros((3, 3)), True),
            ('insufficient_overlap', np.array([[1., 0, 400], [0, 1, 0], [0, 0, 1]]), True),
        ]
        for reason, transform, all_inliers in cases:
            def estimate(source, target, *args, **kwargs):
                return transform, np.full((len(source), 1), int(all_inliers), dtype=np.uint8)
            with self.subTest(reason=reason), patch('data_utils.frame_pair.cv2.findHomography', side_effect=estimate):
                aligned, info = stabilize_past(current, past)
                self.assertFalse(info.success)
                self.assertEqual(info.reason, reason)
                np.testing.assert_array_equal(aligned, past)

    def test_localized_inliers_are_rejected(self):
        points = [cv2.KeyPoint(100. + i % 5, 100. + i // 5, 1) for i in range(20)]
        descriptors = np.random.default_rng(0).random((20, 128), dtype=np.float32)
        with patch('data_utils.frame_pair.cv2.SIFT_create') as sift:
            sift.return_value.detectAndCompute.return_value = (points, descriptors)
            aligned, info = stabilize_past(*textured_pair()[:2])
        self.assertFalse(info.success)
        self.assertEqual(info.reason, 'localized_inliers')

    def test_none_matches_previous_rgb_resize(self):
        current, past, _ = textured_pair()
        images = [Image.fromarray(current), Image.fromarray(past)]
        tensors = prepare_frame_pair(*images, img_size=(96, 64), stabilization='none')
        for image, tensor in zip(images, tensors[:2]):
            expected = torch.from_numpy(np.array(image.convert('RGB').resize(
                (96, 64), Image.Resampling.BILINEAR), dtype=np.float32)).permute(2, 0, 1)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        self.assertEqual(tensors[2].reason, 'disabled')

    def test_dataset_and_inference_share_preprocessing_mask_is_not_warped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'images', root / 'masks'
            images.mkdir(); masks.mkdir()
            current, past, _ = textured_pair()
            Image.fromarray(past).save(images / '0.png')
            Image.fromarray(current).save(images / '1.png')
            mask = np.zeros(current.shape[:2], dtype=np.uint8)
            mask[130:190, 210:250] = 255
            Image.fromarray(mask).save(masks / '1.png')
            dataset = FrameMaskDataset(images, masks, img_size=(320, 240), stabilization='homography')
            current_tensor, past_tensor, target = dataset[0]
            infer_current, infer_past, info, original = load_frame_pair(
                images / '1.png', images / '0.png', (320, 240), 'homography')
            self.assertTrue(info.success, info.summary())
            self.assertEqual(original, (512, 384))
            torch.testing.assert_close(current_tensor, infer_current, rtol=0, atol=0)
            torch.testing.assert_close(past_tensor, infer_past, rtol=0, atol=0)
            expected_current = np.array(Image.fromarray(current).resize((320, 240), Image.Resampling.BILINEAR))
            np.testing.assert_array_equal(current_tensor.permute(1, 2, 0).numpy(), expected_current)
            expected_mask = np.array(Image.fromarray(mask).resize((320, 240), Image.Resampling.NEAREST)) > 0
            np.testing.assert_array_equal(target[0].numpy(), expected_mask)

    def test_warning_once_per_reason_in_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'images', root / 'masks'
            images.mkdir(); masks.mkdir()
            for i in range(3):
                Image.new('RGB', (32, 32), color=(i, i, i)).save(images / f'{i}.png')
                Image.new('L', (32, 32)).save(masks / f'{i}.png')
            dataset = FrameMaskDataset(images, masks, img_size=(32, 32), stabilization='homography')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                dataset[0]; dataset[1]; dataset[0]
            self.assertEqual(len(caught), 1)
            self.assertIn('using unwarped past', str(caught[0].message))


class TinyModel(nn.Module):
    def __init__(self, seg_input='flow'):
        super().__init__()
        self.seg_input = seg_input
        self.backbone_op = nn.Conv2d(3, 3, 1)
        self.seg_head = nn.Conv2d(3, 1, 1)

    def forward(self, current, past, iters_s16=1, iters_s8=8):
        return self.seg_head(self.backbone_op((current - past) / 255))


class ConfigurationChecks(unittest.TestCase):
    def test_stabilization_precedence(self):
        self.assertEqual(resolve_stabilization(None), 'homography')
        self.assertEqual(resolve_stabilization(None, {}), 'none')
        self.assertEqual(resolve_stabilization(None, {'args': Namespace(stabilization='homography')}), 'homography')
        self.assertEqual(resolve_stabilization('none', {'args': {'stabilization': 'homography'}}), 'none')
        self.assertEqual(resolve_stabilization('homography', {}), 'homography')

    def test_inference_configuration_and_overrides(self):
        base = ['--checkpoint', 'model.pth', '--current', 'a.png', '--past', 'b.png', '--output', 'mask.png']
        checkpoint = {'args': Namespace(img_size=[64, 48], iters_s16=2, iters_s8=8, stabilization='homography')}
        args = resolve_config(infer_args(base), checkpoint)
        self.assertEqual((args.img_size, args.iters_s16, args.iters_s8, args.stabilization), ([64, 48], 2, 8, 'homography'))
        args = resolve_config(infer_args(base + ['--iters_s8', '10', '--stabilization', 'none']), checkpoint)
        self.assertEqual((args.iters_s8, args.stabilization), (10, 'none'))
        args = resolve_config(infer_args(base), {})
        self.assertEqual((args.img_size, args.iters_s16, args.iters_s8, args.stabilization), ([960, 544], 1, 8, 'none'))
        for extra in (['--iters_s8', '0'], ['--img_size', '31', '32']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                resolve_config(infer_args(base + extra), checkpoint)
        for extra in (['--threshold', '2'], ['--output', 'mask.jpg']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                infer_args(base + extra)

    @patch('tool.train.ModelBuilder', TinyModel)
    def test_training_saves_and_resumes_stabilization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'images', root / 'masks'
            images.mkdir(); masks.mkdir()
            for i in range(8):
                Image.new('RGB', (32, 32)).save(images / f'{i}.png')
                Image.new('L', (32, 32)).save(masks / f'{i}.png')
            options = ['--data_path', str(root), '--val_split', '.5', '--img_size', '32', '32',
                       '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--no-amp',
                       '--no-tensorboard', '--epochs', '3', '--warm_up_epochs', '0', '--loss', 'BCE',
                       '--backbone_op_pretrained', '', '--work_dir', str(root / 'runs')]
            trainer = Trainer(train_args(options))
            self.assertEqual(trainer.args.stabilization, 'homography')
            self.assertEqual(trainer.train_set.stabilization, 'homography')
            self.assertEqual(trainer.val_set.stabilization, 'homography')
            trainer.save_checkpoint(1)
            path = trainer.save_pth3 / 'last.pth'
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint['args']['stabilization'], 'homography')
            resumed = Trainer(train_args(options + ['--resume', str(path)]))
            self.assertEqual(resumed.train_set.stabilization, 'homography')
            overridden = Trainer(train_args(options + ['--resume', str(path), '--stabilization', 'none']))
            self.assertEqual(overridden.train_set.stabilization, 'none')
            del checkpoint['args']['stabilization']
            old_path = trainer.save_pth3 / 'old.pth'
            torch.save(checkpoint, old_path)
            old = Trainer(train_args(options + ['--resume', str(old_path)]))
            self.assertEqual(old.train_set.stabilization, 'none')


class InferenceChecks(unittest.TestCase):
    def test_real_checkpoint_mask_original_dimensions_at_eight_and_ten(self):
        from NeuFlow.model_builder import ModelBuilder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(42)
            model = ModelBuilder()
            path = root / 'model.pth'
            torch.save({'model': model.state_dict(), 'args': {'img_size': [32, 32],
                       'iters_s8': 8, 'stabilization': 'homography'}}, path)
            current, past = root / 'current.png', root / 'past.png'
            Image.new('RGB', (53, 37), color=(80, 40, 20)).save(current)
            Image.new('RGB', (53, 37), color=(79, 39, 19)).save(past)
            for iterations in (8, 10):
                output = root / 'output' / f'mask_{iterations}.png'
                args = infer_args(['--checkpoint', str(path), '--current', str(current), '--past', str(past),
                                   '--output', str(output), '--device', 'cpu', '--iters_s8', str(iterations)])
                mask, alignment = run_inference(args)
                self.assertEqual(mask.shape, (37, 53))
                self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
                self.assertEqual(alignment.reason, 'insufficient_features')
                with Image.open(output) as image:
                    self.assertEqual((image.size, image.mode, image.format), ((53, 37), 'L', 'PNG'))
                    np.testing.assert_array_equal(np.array(image), mask)

    def test_missing_or_mismatched_input_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, past = root / 'current.png', root / 'past.png'
            Image.new('RGB', (64, 48)).save(current)
            with self.assertRaises(FileNotFoundError):
                load_frame_pair(current, past)
            Image.new('RGB', (32, 48)).save(past)
            with self.assertRaisesRegex(ValueError, 'Frame sizes differ'):
                load_frame_pair(current, past)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
