"""Run with: python checks/check_frame_training.py (PyTorch environment required)."""

import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils.frame_mask_dataset import FrameMaskDataset
from tool.train import Trainer, build_datasets, load_checkpoint, load_weights, parse_args
from external.ILNet.model.metrics import PD_FA


def make_sequence(root, count=12, sequence=''):
    images, masks = root / 'images' / sequence, root / 'masks' / sequence
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        image = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        image[0, 0] = index
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:12, index:index + 5] = 255
        Image.fromarray(image).save(images / f'frame_{index}.png')
        Image.fromarray(mask).save(masks / f'frame_{index}.bmp')
    return images, masks


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, value):
        pass


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.masks = make_sequence(self.root)

    def dataset(self, **options):
        return FrameMaskDataset(self.images, self.masks, img_size=(32, 32), **options)

    def test_current_past_and_current_mask(self):
        dataset = self.dataset(frame_skip=3)
        self.assertEqual(len(dataset), 9)
        current, past, mask = dataset[0]
        self.assertEqual(current[0, 0, 0].item(), 4)
        self.assertEqual(past[0, 0, 0].item(), 1)
        self.assertEqual(dataset.samples[0][2].stem, 'frame_4')
        self.assertTrue(torch.equal(mask[0, 4], torch.tensor([0.] * 4 + [1.] * 5 + [0.] * 23)))
        self.assertEqual(current.shape, (3, 32, 32))
        self.assertEqual(current.dtype, torch.float32)
        self.assertGreater(current.max().item(), 1)
        self.assertEqual(mask.shape, (1, 32, 32))

    def test_natural_sort(self):
        dataset = self.dataset()
        stems = [sample[0].stem for sample in dataset.samples]
        self.assertEqual(stems, [f'frame_{index}' for index in range(2, 13)])

    def test_no_cross_sequence_or_split_overlap(self):
        make_sequence(self.root, sequence='other')
        train = self.dataset(frame_skip=2, split='train', val_split=0.25)
        val = self.dataset(frame_skip=2, split='val', val_split=0.25)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(val), 2)
        used_train = {p for sample in train.samples for p in sample[:2]}
        used_val = {p for sample in val.samples for p in sample[:2]}
        self.assertFalse(used_train & used_val)
        for sample in train.samples + val.samples:
            self.assertEqual(sample[0].parent, sample[1].parent)

    def test_mask_resize_stays_binary(self):
        dataset = FrameMaskDataset(self.images, self.masks, img_size=(64, 48))
        current, past, mask = dataset[0]
        self.assertEqual(current.shape, (3, 48, 64))
        self.assertEqual(past.shape, current.shape)
        self.assertEqual(mask.shape, (1, 48, 64))
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})

    def test_lossless_mask_encodings_and_palette(self):
        mask_path = self.masks / 'frame_2.bmp'
        with Image.open(mask_path) as image:
            original = np.array(image)
        expected = torch.from_numpy((original > 0).astype(np.float32)).unsqueeze(0)
        mask_path.unlink()
        for extension, encoding in [('.bmp', '01'), ('.bmp', '0255'),
                                    ('.png', '01'), ('.png', '0255'), ('.png', 'palette')]:
            with self.subTest(extension=extension, encoding=encoding):
                values = (original > 0).astype(np.uint8)
                if encoding == '0255':
                    values *= 255
                image = Image.fromarray(values)
                if encoding == 'palette':
                    image = image.convert('P')
                    # Background is white, foreground black: read IDs, not palette colors.
                    image.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
                path = self.masks / f'frame_2{extension}'
                image.save(path)
                try:
                    torch.testing.assert_close(self.dataset()[0][2], expected, rtol=0, atol=0)
                finally:
                    path.unlink()

    def test_reject_other_mask_image_formats(self):
        for extension, image_format in [('.jpg', 'JPEG'), ('.jpeg', 'JPEG'),
                                        ('.webp', 'WEBP'), ('.tiff', 'TIFF'), ('.gif', 'GIF')]:
            with self.subTest(extension=extension):
                path = self.masks / f'unsupported{extension}'
                Image.new('L', (32, 32)).save(path, format=image_format)
                try:
                    with self.assertRaisesRegex(ValueError, 'PNG/BMP') as error:
                        self.dataset()
                    self.assertIn(str(path), str(error.exception))
                finally:
                    path.unlink()

    def test_reject_jpeg_disguised_as_png_or_bmp(self):
        (self.masks / 'frame_2.bmp').unlink()
        for extension in ('.png', '.bmp'):
            with self.subTest(extension=extension):
                path = self.masks / f'frame_2{extension}'
                Image.new('L', (32, 32)).save(path, format='JPEG')
                try:
                    with self.assertRaisesRegex(ValueError, 'decoded format is JPEG') as error:
                        self.dataset()[0]
                    self.assertIn(str(path), str(error.exception))
                finally:
                    path.unlink()

    def test_jpeg_input_with_lossless_mask(self):
        expected = self.dataset()[0][2]
        for index in (1, 2):
            path = self.images / f'frame_{index}.png'
            with Image.open(path) as image:
                image.save(path.with_suffix('.jpg'))
            path.unlink()
        dataset = self.dataset()
        current, past, mask = dataset[0]
        self.assertEqual([path.suffix for path in dataset.samples[0][:2]], ['.jpg', '.jpg'])
        self.assertEqual(current.shape, (3, 32, 32))
        self.assertEqual(past.shape, current.shape)
        torch.testing.assert_close(mask, expected, rtol=0, atol=0)

    def test_missing_target_mask_and_bad_skip(self):
        with self.assertRaisesRegex(ValueError, 'frame_skip'):
            self.dataset(frame_skip=0)
        (self.masks / 'frame_1.bmp').unlink()  # A past-only frame needs no mask.
        self.dataset()
        (self.masks / 'frame_2.bmp').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'frame_2'):
            self.dataset()

    def test_duplicate_stem_and_dimension_errors(self):
        Image.new('L', (32, 32)).save(self.masks / 'frame_2.png')
        with self.assertRaisesRegex(ValueError, 'Ambiguous mask'):
            self.dataset()
        (self.masks / 'frame_2.png').unlink()
        Image.new('L', (16, 32)).save(self.masks / 'frame_2.bmp')
        with self.assertRaisesRegex(ValueError, 'sizes differ'):
            self.dataset()[0]

    def test_cli_and_explicit_validation_overlap(self):
        args = parse_args(['--data_path', str(self.root), '--frame_skip', '2', '--img_size', '64', '32'])
        self.assertEqual(args.image_dir, str(self.images))
        self.assertEqual(args.img_size, [64, 32])
        self.assertEqual(args.frame_skip, 2)
        self.assertEqual(args.epoch_training_backbone, 300)
        self.assertEqual((args.iters_s16, args.iters_s8), (1, 8))
        args.val_image_dir, args.val_mask_dir = str(self.images), str(self.masks)
        with self.assertRaisesRegex(ValueError, 'same frame paths'):
            build_datasets(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--data_path', str(self.root), '--frame_skip', '0'])

    def test_iteration_cli(self):
        args = parse_args(['--data_path', str(self.root), '--iters_s16', '2', '--iters_s8', '10'])
        self.assertEqual((args.iters_s16, args.iters_s8), (2, 10))
        for option in ('--iters_s16', '--iters_s8'):
            for value in ('0', '-1', '1.5'):
                with self.subTest(option=option, value=value):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_args(['--data_path', str(self.root), option, value])


class TinyModel(nn.Module):
    def __init__(self, seg_input='flow'):
        super().__init__()
        self.seg_input = seg_input
        self.backbone_op = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.seg_head = nn.Conv2d(3, 1, 1)

    def forward(self, current, past, iters_s16=1, iters_s8=8):
        self.last_iterations = (iters_s16, iters_s8)
        return self.seg_head(self.backbone_op(current - past))


class MetricChecks(unittest.TestCase):
    def test_perfect_batch_detection_and_equal_area_false_alarm(self):
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 2:4, 2:4] = 1
        prediction = target * 2 - 1
        metric = PD_FA((16, 16))
        metric.update(prediction, target)
        fa, pd = metric.get(2)
        self.assertEqual(metric.target, 2)
        self.assertEqual(pd, 1.0)
        self.assertEqual(fa, 0.0)
        metric.reset()
        prediction[0, :, 10:12, 10:12] = 1
        metric.update(prediction, target)
        fa, pd = metric.get(2)
        self.assertEqual(pd, 1.0)
        self.assertEqual(fa, 4 / (2 * 16 * 16))


class OptimizerChecks(unittest.TestCase):
    @patch('tool.train.tqdm', QuietProgress)
    @patch('tool.train.ModelBuilder', TinyModel)
    def test_training_validation_and_teacher_receive_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sequence(root, count=6)
            teacher_path = root / 'teacher.pth'
            torch.save({'model': TinyModel().state_dict()}, teacher_path)
            args = parse_args(['--data_path', str(root), '--val_split', '0.5',
                               '--img_size', '32', '32', '--batch_size', '2', '--num_workers', '0',
                               '--device', 'cpu', '--no-amp', '--no-tensorboard',
                               '--epochs', '2', '--warm_up_epochs', '0', '--loss', 'BCE',
                               '--backbone_op_pretrained', '', '--work_dir', str(root / 'runs'),
                               '--iters_s16', '2', '--iters_s8', '10', '--teacher', str(teacher_path)])
            trainer = Trainer(args)
            self.assertTrue(math.isfinite(trainer.training(1)))
            self.assertEqual(trainer.model.last_iterations, (2, 10))
            self.assertEqual(trainer.teacher.last_iterations, (2, 10))
            trainer.model.last_iterations = None
            trainer.validation(1)
            self.assertEqual(trainer.model.last_iterations, (2, 10))

    @patch('tool.train.tqdm', QuietProgress)
    def test_freeze_and_lr_for_both_optimizers_and_schedules(self):
        for optimizer_name in ('adan', 'msam'):
            for schedule in ('ilnet', 'cos'):
                with self.subTest(optimizer=optimizer_name, schedule=schedule):
                    trainer = Trainer.__new__(Trainer)
                    trainer.args = SimpleNamespace(dataset='frames', optim=optimizer_name,
                        lr_sch=schedule, learning_rate=1e-3, backbone_layer_lr=0.1,
                        epoch_training_backbone=3, warm_up_epochs=1, epochs=5,
                        loss='BCE', clip_grad=0.0, iters_s16=2, iters_s8=10)
                    trainer.model = TinyModel()
                    trainer.device = torch.device('cpu')
                    trainer.teacher = trainer.writer = None
                    trainer.scaler = torch.amp.GradScaler('cuda', enabled=False)
                    trainer.criterion = nn.BCEWithLogitsLoss()
                    trainer.train_data_loader = [(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4),
                                                  torch.ones(2, 1, 4, 4))]
                    trainer.optimizer, trainer.lr_scheduler = trainer.build_opt_lr(trainer.model)
                    original_optimizer, original_scheduler = trainer.optimizer, trainer.lr_scheduler
                    before = {key: value.clone() for key, value in trainer.model.backbone_op.state_dict().items()}
                    for epoch in range(1, 6):
                        trainer.training(epoch)
                        self.assertEqual(trainer.model.last_iterations, (2, 10))
                        self.assertIs(trainer.optimizer, original_optimizer)
                        self.assertIs(trainer.lr_scheduler, original_scheduler)
                        head_lr, backbone_lr = [group['lr'] for group in trainer.optimizer.param_groups]
                        self.assertTrue(math.isclose(backbone_lr, head_lr * 0.1, rel_tol=1e-12))
                        self.assertEqual(trainer.model.backbone_op.training, epoch >= 3)
                        for parameter in trainer.model.backbone_op.parameters():
                            self.assertEqual(parameter.requires_grad, epoch >= 3)
                        if epoch < 3:
                            for key, expected in before.items():
                                self.assertTrue(torch.equal(expected, trainer.model.backbone_op.state_dict()[key]))
                        else:
                            self.assertFalse(torch.equal(before['0.weight'], trainer.model.backbone_op[0].weight))
                    # Re-freezing a resumed MSAM state must not perturb frozen weights.
                    trainer.args.epoch_training_backbone = 10
                    trainer._set_backbone_training(5)
                    before = [p.clone() for p in trainer.model.backbone_op.parameters()]
                    if optimizer_name == 'msam':
                        trainer.optimizer.move_up_to_momentumAscent()
                        for old, new in zip(before, trainer.model.backbone_op.parameters()):
                            self.assertTrue(torch.equal(old, new))
                        trainer.optimizer.move_back_from_momentumAscent()


class RealModelChecks(unittest.TestCase):
    def test_mask_loss_trains_flow_stages_with_all_iterations(self):
        self._check_refinement_gradients(1, 8, use_defaults=True)

    def test_mask_loss_trains_multiple_s16_and_ten_s8_iterations(self):
        self._check_refinement_gradients(2, 10)

    def _check_refinement_gradients(self, iters_s16, iters_s8, use_defaults=False):
        from NeuFlow.model_builder import ModelBuilder

        torch.manual_seed(42)
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(loss='BCE', dataset='frames',
                                       epoch_training_backbone=300, backbone_layer_lr=0.1,
                                       iters_s16=iters_s16, iters_s8=iters_s8)
        trainer.model = ModelBuilder().train()
        trainer.criterion = trainer.build_loss()
        trainer._set_backbone_training(300)

        backbone = trainer.model.backbone_op
        counts = dict(s16=0, s8=0, image_feature=0, upsampling=0)
        for name, module in [('s16', backbone.refine_s16), ('s8', backbone.refine_s8),
                             ('image_feature', backbone.conv_s8), ('upsampling', backbone.upsample_s8)]:
            def count_calls(module, inputs, output, name=name):
                counts[name] += 1
            handle = module.register_forward_hook(count_calls)
            self.addCleanup(handle.remove)
        flow_prediction_counts = []
        handle = backbone.register_forward_hook(
            lambda module, inputs, output: flow_prediction_counts.append(len(output)))
        self.addCleanup(handle.remove)
        deltas = dict(s16=[], s8=[])
        for name, module in [('s16', backbone.refine_s16), ('s8', backbone.refine_s8)]:
            def retain_delta(module, inputs, output, name=name):
                delta = output[1]
                delta.retain_grad()
                deltas[name].append(delta)
            handle = module.register_forward_hook(retain_delta)
            self.addCleanup(handle.remove)

        current = torch.rand(2, 3, 32, 32) * 255
        past = torch.rand_like(current) * 255
        mask = (torch.rand(2, 1, 32, 32) > 0.8).float()
        output = (trainer.model(current, past) if use_defaults
                  else trainer._forward(trainer.model, [current, past]))
        # Mask ground truth is the only supervision supplied to the actual trainer loss.
        loss = trainer._loss(output, mask, epoch=300)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        self.assertEqual(counts, dict(s16=iters_s16, s8=iters_s8,
                                     image_feature=iters_s8, upsampling=iters_s8))
        self.assertEqual(flow_prediction_counts, [iters_s16 + iters_s8])
        self.assertEqual(backbone.conv_s8.norm1.num_batches_tracked.item(), iters_s8)
        for name, iterations in [('s16', iters_s16), ('s8', iters_s8)]:
            self.assertEqual(len(deltas[name]), iterations)
            for index, delta in enumerate(deltas[name]):
                with self.subTest(stage=name, iteration=index + 1):
                    self.assertIsNotNone(delta.grad)
                    self.assertTrue(torch.isfinite(delta.grad).all().item())
                    self.assertGreater(torch.count_nonzero(delta.grad).item(), 0)
        for name, module in [('CNN', backbone.backbone), ('S16', backbone.refine_s16),
                             ('S8', backbone.refine_s8), ('upsampling', backbone.upsample_s8),
                             ('segmentation', trainer.model.seg_head)]:
            with self.subTest(module=name):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
                self.assertTrue(any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients))

    def test_neuflow_keeps_legacy_detach_default(self):
        from NeuFlow.neuflow import NeuFlow

        torch.manual_seed(42)
        backbone = NeuFlow().train()
        backbone.init_bhwd(2, 32, 32, torch.device('cpu'), amp=False)
        deltas = []
        def retain_delta(module, inputs, output):
            delta = output[1]
            delta.retain_grad()
            deltas.append(delta)
        handle = backbone.refine_s8.register_forward_hook(retain_delta)
        self.addCleanup(handle.remove)
        flow = backbone(torch.rand(2, 3, 32, 32) * 255, torch.rand(2, 3, 32, 32) * 255)[-1]
        flow.square().mean().backward()
        self.assertEqual(len(deltas), 8)
        self.assertTrue(all(delta.grad is None for delta in deltas[:-1]))
        self.assertIsNotNone(deltas[-1].grad)
        self.assertTrue(torch.isfinite(deltas[-1].grad).all().item())
        self.assertGreater(torch.count_nonzero(deltas[-1].grad).item(), 0)

    def test_invalid_iteration_api(self):
        from NeuFlow.model_builder import ModelBuilder

        model = ModelBuilder().eval()
        image = torch.zeros(1, 3, 32, 32)
        for name in ('iters_s16', 'iters_s8'):
            for value in (0, -1, 1.5, True, '8'):
                with self.subTest(option=name, value=value):
                    with self.assertRaisesRegex(ValueError, f'{name} must be a positive integer'):
                        model(image, image, **{name: value})

    @patch('tool.train.tqdm', QuietProgress)
    def test_checkpoint_trained_eight_can_infer_ten(self):
        from NeuFlow.model_builder import ModelBuilder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sequence(root, count=3)
            options = ['--data_path', str(root), '--val_split', '0', '--img_size', '32', '32',
                       '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--no-amp',
                       '--no-tensorboard', '--epochs', '2', '--warm_up_epochs', '0', '--loss', 'BCE',
                       '--epoch_training_backbone', '1', '--backbone_op_pretrained', '',
                       '--work_dir', str(root / 'runs')]
            torch.manual_seed(42)
            trainer = Trainer(parse_args(options))
            self.assertTrue(math.isfinite(trainer.training(1)))
            trainer.save_checkpoint(1)
            path = trainer.save_pth3 / 'last.pth'
            checkpoint = load_checkpoint(path)
            self.assertEqual((checkpoint['args']['iters_s16'], checkpoint['args']['iters_s8']), (1, 8))
            inference_model = ModelBuilder().eval()
            load_weights(inference_model, path)
            counts = dict(s16=0, s8=0, image_feature=0, upsampling=0)
            for name, module in [('s16', inference_model.backbone_op.refine_s16),
                                 ('s8', inference_model.backbone_op.refine_s8),
                                 ('image_feature', inference_model.backbone_op.conv_s8),
                                 ('upsampling', inference_model.backbone_op.upsample_s8)]:
                def count_calls(module, inputs, output, name=name):
                    counts[name] += 1
                handle = module.register_forward_hook(count_calls)
                self.addCleanup(handle.remove)
            flow_counts = []
            handle = inference_model.backbone_op.register_forward_hook(
                lambda module, inputs, output: flow_counts.append(len(output)))
            self.addCleanup(handle.remove)
            current, past, _ = trainer.train_set[0]
            with torch.inference_mode():
                prediction = inference_model(current.unsqueeze(0), past.unsqueeze(0), iters_s8=10)
            self.assertEqual(counts, dict(s16=1, s8=10, image_feature=1, upsampling=1))
            self.assertEqual(flow_counts, [1])
            self.assertEqual(prediction.shape, (1, 1, 32, 32))
            self.assertTrue(torch.isfinite(prediction).all().item())
            self.assertEqual(inference_model.backbone_op.conv_s8.norm1.num_batches_tracked.item(), 8)
            for key, value in inference_model.state_dict().items():
                torch.testing.assert_close(value, checkpoint['model'][key], rtol=0, atol=0, msg=key)

            # Earlier checkpoints have no iteration metadata; CLI defaults are still 1/8.
            del checkpoint['args']['iters_s16'], checkpoint['args']['iters_s8']
            legacy_path = trainer.save_pth3 / 'old_format.pth'
            torch.save(checkpoint, legacy_path)
            resumed = Trainer(parse_args(options + ['--resume', str(legacy_path)]))
            self.assertEqual((resumed.args.iters_s16, resumed.args.iters_s8), (1, 8))
            torch.manual_seed(99)
            self.assertTrue(math.isfinite(trainer.training(2)))
            torch.manual_seed(99)
            self.assertTrue(math.isfinite(resumed.training(2)))
            for key, expected in trainer.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], expected, rtol=0, atol=0, msg=key)

    @patch('tool.train.tqdm', QuietProgress)
    def test_images_to_masks_freeze_299_unfreeze_300_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sequence(root, count=8)
            options = ['--data_path', str(root), '--val_split', '0.5', '--img_size', '32', '32',
                       '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--no-amp',
                       '--no-tensorboard', '--epochs', '301', '--warm_up_epochs', '0',
                       '--work_dir', str(root / 'runs')]
            torch.manual_seed(17)
            trainer = Trainer(parse_args(options))
            backbone_before = {key: value.clone() for key, value in trainer.model.backbone_op.state_dict().items()}
            head_before = trainer.model.seg_head.out_cnn.weight.detach().clone()
            current, past, _ = next(iter(trainer.train_data_loader))
            current_copy, past_copy = current.clone(), past.clone()
            with torch.no_grad():
                output = trainer.model(current, past)
            self.assertEqual(output[0].shape, (2, 1, 32, 32))
            self.assertTrue(torch.equal(current, current_copy))
            self.assertTrue(torch.equal(past, past_copy))
            self.assertTrue(math.isfinite(trainer.training(299)))
            self.assertFalse(trainer.model.backbone_op.training)
            for key, expected in backbone_before.items():
                self.assertTrue(torch.equal(expected, trainer.model.backbone_op.state_dict()[key]), key)
            self.assertFalse(torch.equal(head_before, trainer.model.seg_head.out_cnn.weight))
            improved = trainer.validation(299)  # Includes a final validation batch of size one.
            trainer.save_checkpoint(299, improved)
            self.assertTrue((trainer.save_pth3 / 'last.pth').is_file())
            self.assertFalse(list(trainer.save_pth3.glob('*.tmp')))
            self.assertTrue(math.isfinite(trainer.training(300)))
            self.assertTrue(trainer.model.backbone_op.training)
            self.assertTrue(any(not torch.equal(old, trainer.model.backbone_op.state_dict()[key])
                                for key, old in backbone_before.items() if key.endswith('weight')))
            resumed = Trainer(parse_args(options + ['--resume', str(trainer.save_pth3 / 'last.pth')]))
            self.assertEqual(resumed.args.start_epoch, 300)
            self.assertTrue(resumed.backbone_training)
            self.assertEqual(resumed.optimizer.param_groups[0]['step'], 1)
            self.assertTrue(math.isfinite(resumed.training(300)))
            for key, expected in trainer.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], expected, rtol=0, atol=0, msg=key)
            self.assertAlmostEqual(resumed.optimizer.param_groups[1]['lr'], resumed.optimizer.param_groups[0]['lr'] * 0.1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
