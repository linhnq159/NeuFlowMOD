"""Joint augmentations, component preservation and training/checkpoint integration."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from NeuFlow import config
from checks.check_frame_training import make_sequence, QuietProgress, TinyModel
from data_utils.frame_augmentation import FrameMaskAugmentor, validate_augmentation
from data_utils.frame_mask_dataset import FrameMaskDataset
from data_utils.frame_pair import init_frame_worker, load_frame_pair
from tool.checkpoints import load_checkpoint, resolve_train_augmentation
from tool.train import build_datasets, parse_args, Trainer


def settings(**changes):
    result = deepcopy(config.DEFAULT_TRAIN_AUGMENTATION)
    result.update(changes)
    return result


def numpy_batches(samples):
    # Test worker RNG via ordinary pickled arrays. Unlike tensor FD sharing,
    # this transport also works in sandboxes that prohibit local IPC sockets.
    return tuple(torch.stack(values).numpy() for values in zip(*samples))


class GeometryChecks(unittest.TestCase):
    def setUp(self):
        self.current = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8).expand(3, -1, -1).clone()
        self.past = self.current + 20
        self.mask = torch.zeros(1, 8, 8)
        self.mask[:, 2:6, 2:6] = 1

    def apply_scale(self, factor, mask=None):
        with patch('data_utils.frame_augmentation._offset', return_value=0):
            return FrameMaskAugmentor(settings())._scale(
                self.current, self.past, self.mask if mask is None else mask, factor)

    def test_disabled_exact_identity_and_no_random_draw(self):
        state = torch.get_rng_state()
        result = FrameMaskAugmentor(settings())(self.current, self.past, self.mask)
        for actual, expected in zip(result, (self.current, self.past, self.mask)):
            self.assertIs(actual, expected)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_horizontal_flip_and_shared_photometric_parameters(self):
        aug = FrameMaskAugmentor(settings(enabled=['hflip', 'brightness_contrast'], hflip_p=1,
                                         brightness_contrast_p=1, brightness_range=[1.1, 1.1],
                                         contrast_range=[0.9, 0.9]))
        before = [v.clone() for v in (self.current, self.past, self.mask)]
        current, past, mask = aug(self.current, self.past, self.mask)
        self.assertTrue(torch.equal(mask, self.mask.flip(-1)))
        torch.testing.assert_close(current, ((self.current.flip(-1) - 127.5) * .9 + 127.5) * 1.1)
        torch.testing.assert_close(past - current, torch.full_like(current, 19.8))
        for actual, expected in zip((self.current, self.past, self.mask), before):
            self.assertTrue(torch.equal(actual, expected))

    def test_photometric_clamps_and_does_not_change_mask(self):
        aug = FrameMaskAugmentor(settings(enabled=['brightness_contrast'], brightness_contrast_p=1,
                                         brightness_range=[2, 2], contrast_range=[2, 2]))
        current, past, mask = aug(self.current, torch.full_like(self.past, 255), self.mask)
        self.assertEqual(current.min().item(), 0)
        self.assertEqual(past.max().item(), 255)
        self.assertIs(mask, self.mask)

    def test_zoom_out_shared_canvas_position_color_and_binary_mask(self):
        current, past, mask = self.apply_scale(.5)
        expected = torch.zeros_like(mask)
        expected[:, 1:3, 1:3] = 1
        self.assertTrue(torch.equal(mask, expected))
        self.assertEqual(current.shape, self.current.shape)
        torch.testing.assert_close(past[:, :4, :4] - current[:, :4, :4], torch.full((3, 4, 4), 20.))
        self.assertTrue(torch.equal(current[:, 4:, :], past[:, 4:, :]))
        self.assertEqual(current[0, 7, 7].item(), 41.5)

    def test_zoom_in_partial_region_is_allowed(self):
        current, past, mask = self.apply_scale(2)
        self.assertFalse(torch.equal(current, self.current))
        self.assertEqual(mask.sum().item(), 16)
        self.assertTrue((mask[:, 4:, 4:] == 1).all())
        torch.testing.assert_close(past - current, torch.full_like(current, 20.))

    def test_image_and_mask_object_share_random_padding_offset(self):
        current = self.mask.expand(3, -1, -1) * 100
        with patch('data_utils.frame_augmentation._offset', return_value=2):
            first, second, mask = FrameMaskAugmentor(settings())._scale(current, current + 10, self.mask, .5)
        expected = torch.zeros_like(mask)
        expected[:, 3:5, 3:5] = 1
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue((first[:, 3:5, 3:5] == 100).all())
        self.assertTrue((second[:, 3:5, 3:5] == 110).all())

    def test_rounding_on_odd_sizes_and_surviving_single_pixel(self):
        aug = FrameMaskAugmentor(settings())
        current, past = torch.zeros(3, 35, 47), torch.ones(3, 35, 47)
        for factor in (.6, .999, 1.001, 1.2):
            result = aug._scale(current, past, torch.zeros(1, 35, 47), factor)
            self.assertEqual([tuple(value.shape[-2:]) for value in result], [(35, 47)] * 3)
        mask = torch.zeros_like(self.mask)
        mask[:, 1, 1] = 1
        first, _, target = self.apply_scale(.5, mask)
        self.assertEqual(target.sum(), 1)
        self.assertIsNot(first, self.current)

    def test_missing_one_of_two_regions_cancels_crop(self):
        mask = torch.zeros_like(self.mask)
        mask[:, 1:3, 1:3] = 1
        mask[:, 6:8, 6:8] = 1
        result = self.apply_scale(2, mask)
        for actual, expected in zip(result, (self.current, self.past, mask)):
            self.assertIs(actual, expected)

    def test_single_pixel_disappears_on_downsample(self):
        mask = self.mask.clone()
        mask[:, 0, 0] = 1  # nearest-exact at scale .5 samples indices 1,3,5,7
        result = self.apply_scale(.5, mask)
        for actual, expected in zip(result, (self.current, self.past, mask)):
            self.assertIs(actual, expected)

    def test_eight_connected_region_can_survive_with_only_part_remaining(self):
        mask = torch.zeros_like(self.mask)
        mask[0].fill_diagonal_(1)
        current, _, target = self.apply_scale(2, mask)
        self.assertFalse(torch.equal(current, self.current))
        self.assertGreater(target.sum(), 0)

    def test_empty_masks_can_scale_and_preserve_shape(self):
        for factor in (.6, 1.2):
            with self.subTest(factor=factor):
                current, past, target = self.apply_scale(factor, torch.zeros_like(self.mask))
                self.assertEqual(current.shape, self.current.shape)
                self.assertEqual(past.shape, self.past.shape)
                self.assertEqual(target.shape, self.mask.shape)
                self.assertEqual(target.sum(), 0)
                self.assertFalse(torch.equal(current, self.current))

    def test_scale_fallback_keeps_flip_and_photometric(self):
        mask = self.mask.clone()
        mask[:, 0, 0] = 1
        aug = FrameMaskAugmentor(settings(enabled=['scale', 'hflip', 'brightness_contrast'],
                                         zoom_out_p=1, zoom_in_p=0, zoom_out_range=[.5, .5],
                                         hflip_p=1, brightness_contrast_p=1,
                                         brightness_range=[.9, .9], contrast_range=[1, 1]))
        current, past, target = aug(self.current, self.past, mask)
        torch.testing.assert_close(current, self.current.flip(-1) * .9)
        torch.testing.assert_close(past, self.past.flip(-1) * .9)
        self.assertTrue(torch.equal(target, mask.flip(-1)))

    def test_scale_branch_boundaries_and_relative_probability(self):
        aug = FrameMaskAugmentor(settings(enabled=['scale']))
        for draw, expected in ((0., 'out'), (.399, 'out'), (.4, 'in'), (.499, 'in'), (.5, None), (.99, None)):
            with patch('torch.rand', return_value=torch.tensor(draw)), \
                    patch.object(aug, '_scale', return_value=(self.current, self.past, self.mask)) as scale:
                aug(self.current, self.past, self.mask)
                self.assertEqual(scale.call_count, int(expected is not None))
                if expected:
                    self.assertEqual(scale.call_args.args[3] < 1, expected == 'out')


class ConfigurationChecks(unittest.TestCase):
    def test_invalid_settings(self):
        cases = [dict(hflip_p=float('nan')), dict(zoom_out_p=1.1), dict(zoom_in_p=.4),
                 dict(zoom_out_p=.8, zoom_in_p=.3), dict(zoom_out_p=0),
                 dict(zoom_out_range=[.5, 1.1]), dict(zoom_in_range=[.9, 1.2]),
                 dict(brightness_range=[1, .9]), dict(contrast_range=[0, 1]),
                 dict(contrast_range=[1, float('inf')]), dict(enabled=['noise']),
                 dict(enabled=['hflip', 'hflip']), dict(enabled='hflip'), dict(unknown=1)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_augmentation(settings(**case))
        validate_augmentation(settings(zoom_in_p=0))
        validate_augmentation(settings(zoom_out_p=0, zoom_in_p=0))

    def test_cli_none_and_invalid_combinations(self):
        self.assertEqual(parse_args(['--data_path', 'unused', '--augment', 'none']).augment, ['none'])
        for extra in (['none', 'scale'], ['scale', 'scale'], ['noise']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--data_path', 'unused', '--augment', *extra])

    def test_config_resume_legacy_and_explicit_override(self):
        custom = settings(enabled=['scale'], zoom_out_p=.7, zoom_in_p=.2)
        with patch.object(config, 'train_augmentation', custom):
            self.assertEqual(resolve_train_augmentation(), custom)
            self.assertEqual(resolve_train_augmentation(checkpoint={})['enabled'], [])
            saved = {'args': {'train_augmentation': settings(enabled=['hflip'], hflip_p=.9)}}
            self.assertEqual(resolve_train_augmentation(checkpoint=saved)['hflip_p'], .9)
            self.assertEqual(resolve_train_augmentation(['scale'], saved)['enabled'], ['scale'])
            self.assertEqual(resolve_train_augmentation(['none'], saved)['enabled'], [])
            self.assertEqual(resolve_train_augmentation(['scale'], saved)['hflip_p'], .9)
        with self.assertRaises(ValueError):
            resolve_train_augmentation(['none', 'hflip'])


class IntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.masks = make_sequence(self.root, count=12)

    def args(self, *extra):
        return parse_args(['--data_path', str(self.root), '--resize', '32', '32', '--stabilization', 'none',
                           '--epochs', '2', '--warm_up_epochs', '0', '--batch_size', '2',
                           '--num_workers', '0', '--device', 'cpu', '--no-amp', '--no-tensorboard',
                           '--backbone_op_pretrained', '', '--work_dir', str(self.root / 'runs'), *extra])

    def test_validation_is_never_augmented_for_both_split_styles(self):
        images, masks = make_sequence(self.root / 'validation', count=6)
        for extra in ([], ['--val_image_dir', str(images), '--val_mask_dir', str(masks)]):
            args = self.args('--augment', 'hflip', *extra)
            with patch.object(config, 'train_augmentation', settings(hflip_p=1)):
                train, val = build_datasets(args)
            self.assertEqual(train.augmentor.settings['enabled'], ['hflip'])
            self.assertIsNone(val.augmentor)
            baseline, _ = build_datasets(self.args('--augment', 'none', *extra))
            for actual, original in zip(train[0], baseline[0]):
                self.assertTrue(torch.equal(actual, original.flip(-1)))
            state = torch.get_rng_state()
            first, second = val[0], val[0]
            for a, b in zip(first, second):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_native_size_and_inference_preprocessing_are_unchanged(self):
        dataset = FrameMaskDataset(self.images, self.masks,
                                   augmentation=settings(enabled=['hflip'], hflip_p=1))
        source, past, _ = dataset.samples[0]
        current, reference, _, _ = load_frame_pair(source, past)
        augmented = dataset[0]
        self.assertTrue(torch.equal(augmented[0], current.flip(-1)))
        self.assertTrue(torch.equal(augmented[1], reference.flip(-1)))

    def test_seed_reproducibility_with_zero_and_multiple_workers(self):
        dataset = FrameMaskDataset(self.images, self.masks, augmentation=settings(
            enabled=['hflip', 'scale', 'brightness_contrast']))
        for workers in (0, 2):
            def batches(seed):
                torch.manual_seed(seed)
                return list(DataLoader(dataset, batch_size=3, num_workers=workers,
                                       worker_init_fn=init_frame_worker, collate_fn=numpy_batches,
                                       timeout=15 if workers else 0))
            first, second, different = batches(111), batches(111), batches(112)
            for a, b in zip(first, second):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(torch.from_numpy(x), torch.from_numpy(y)))
            self.assertTrue(any(not torch.equal(torch.from_numpy(a[0]), torch.from_numpy(b[0]))
                                for a, b in zip(first, different)))

    def test_checkpoint_logs_resume_and_rng_round_trip(self):
        with patch('tool.train.ModelBuilder', TinyModel), patch('tool.train.tqdm', QuietProgress):
            trainer = Trainer(self.args('--augment', 'hflip', 'scale', 'brightness_contrast'))
            self.addCleanup(trainer.close)
            trainer.training(1)
            trainer.save_checkpoint(1, trainer.validation(1))
            path = trainer.save_pth3 / 'last.pth'
            saved = load_checkpoint(path)  # weights_only=True, including on torch 2.3
            expected_config = trainer.args.train_augmentation
            self.assertEqual(saved['args']['train_augmentation'], expected_config)
            expected_next = trainer.train_set[0]
            for path_config in (trainer.save_folder3 / 'config.json', trainer.history.session_dir / 'config.json'):
                self.assertEqual(json.loads(path_config.read_text())['train_augmentation'], expected_config)
            self.assertIn('train_augmentation', (trainer.save_folder3 / 'train.log').read_text())
            with patch.object(config, 'train_augmentation', settings()):
                resumed = Trainer(self.args('--resume', str(path)))
            self.addCleanup(resumed.close)
            self.assertEqual(resumed.args.train_augmentation, expected_config)
            for actual, expected in zip(resumed.train_set[0], expected_next):
                self.assertTrue(torch.equal(actual, expected))
            resumed.training(2)
            resumed.save_checkpoint(2, resumed.validation(2))
            self.assertEqual(len((trainer.save_folder3 / 'history.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    unittest.main(verbosity=2)
