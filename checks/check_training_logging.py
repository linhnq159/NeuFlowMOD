"""CUSTOM gradients, durable histories, resume and TensorBoard previews.

Run: python checks/check_training_logging.py (PyTorch 2.3+).
"""

import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.check_frame_training import TinyModel, QuietProgress, make_sequence
from data_utils.frame_mask_dataset import FrameMaskDataset
from external.ILNet.loss.custom_loss import CustomLoss, CUSTOM_LOSS_VERSION
from tool.checkpoints import load_checkpoint
from tool.train import Trainer, parse_args, run_training
from tool.training_log import SegmentationStats, VAL_FIELDS, preview_panel, select_preview_samples


class RecordingWriter:
    def __init__(self):
        self.images, self.scalars = [], []

    def add_image(self, tag, image, step):
        self.images.append((tag, image.clone(), step))

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class AuxiliaryModel(TinyModel):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return (result, result * 0.5, result * 0.25) if self.training else result


class LossChecks(unittest.TestCase):
    def test_low_probability_foreground_gradient_and_unchanged_normalization(self):
        logits = torch.tensor([-7.0, -4.0, 2.0, 0.0], requires_grad=True)
        target = torch.tensor([1., 1., 0., 0.])
        probability = logits.sigmoid()
        expected = 1 - ((2 * (probability * target).sum() + 1)
                        / ((probability**2).sum() + target.sum() + 1)).sqrt()
        loss = CustomLoss()(logits, target, threshold=0.99)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue((logits.grad[:2] < 0).all())
        self.assertTrue((logits.grad[2:] > 0).all())
        torch.testing.assert_close(loss, CustomLoss()(logits, target, threshold=0))

    def test_empty_sparse_targets_and_all_auxiliary_gradients(self):
        for empty in (True, False):
            outputs = [torch.full((2, 1, 32, 32), -6., requires_grad=True) for _ in range(4)]
            target = torch.zeros_like(outputs[0])
            if not empty:
                target[0, 0, 4, 5] = 1
            trainer = Trainer.__new__(Trainer)
            trainer.args = SimpleNamespace(loss='CUSTOM')
            trainer.criterion = trainer.build_loss()
            loss = trainer._loss(outputs, target, 1)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for output in outputs:
                self.assertTrue(torch.isfinite(output.grad).all())
                self.assertGreater(output.grad.abs().sum().item(), 0)
                if not empty:
                    self.assertLess(output.grad[0, 0, 4, 5].item(), 0)

    def test_real_model_custom_keeps_every_auxiliary_output_differentiable(self):
        from NeuFlow.model_builder import ModelBuilder

        torch.manual_seed(42)
        model = ModelBuilder().train()
        current = torch.rand(2, 3, 32, 32) * 255
        outputs = model(current, torch.rand_like(current) * 255)
        self.assertGreater(len(outputs), 1)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 4:6, 7:9] = 1
        for output in outputs:
            output.retain_grad()
        CustomLoss()(outputs, target).backward()
        for output in outputs:
            self.assertIsNotNone(output.grad)
            self.assertTrue(torch.isfinite(output.grad).all())
            self.assertGreater(output.grad.abs().sum().item(), 0)

    def test_confusion_metrics_weight_pixels_and_positive_only_niou(self):
        metric = SegmentationStats()
        metric.update(torch.tensor([[[[1., 1.], [-1., -1.]]]]),
                      torch.tensor([[[[1., 0.], [0., 0.]]]]))
        metric.update(torch.full((2, 1, 1, 3), -1.), torch.zeros(2, 1, 1, 3))
        values = metric.values()
        for name, expected in dict(iou=0.5, precision=0.5, recall=1, niou=1 / 6,
                                   niou_positive=0.5, target_fg_ratio=0.1,
                                   pred_fg_ratio=0.2, positive_samples=1).items():
            self.assertAlmostEqual(values[name], expected)
        self.assertIsNone(SegmentationStats().values()['niou_positive'])


class TrainingLogChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_sequence(self.root, count=10)
        self.model_patch = patch('tool.train.ModelBuilder', AuxiliaryModel)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.progress_patch = patch('tool.train.tqdm', QuietProgress)
        self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)

    def args(self, *extra):
        return parse_args(['--data_path', str(self.root), '--resize', '32', '32',
                           '--batch_size', '2', '--num_workers', '0', '--device', 'cpu',
                           '--no-amp', '--no-tensorboard', '--epochs', '4', '--warm_up_epochs', '0',
                           '--backbone_op_pretrained', '', '--stabilization', 'none',
                           '--val_split', '0.5', '--work_dir', str(self.root / 'runs'), *extra])

    def trainer(self, *extra):
        trainer = Trainer(self.args(*extra))
        self.addCleanup(trainer.close)
        return trainer

    def rows(self, trainer):
        return [json.loads(line) for line in (trainer.save_folder3 / 'history.jsonl').read_text().splitlines()]

    def epoch(self, trainer, epoch, validate=True):
        trainer.training(epoch)
        improved = trainer.validation(epoch) if validate else []
        trainer.save_checkpoint(epoch, improved)

    def test_epoch_files_checkpoint_and_sample_weighted_main_aux_total(self):
        trainer = self.trainer('--loss', 'BCE')
        # Unequal batch sizes: independently measure loss/metrics before updates.
        data = [(torch.randn(size, 3, 4, 4), torch.randn(size, 3, 4, 4),
                 (torch.rand(size, 1, 4, 4) > 0.8).float()) for size in (2, 3)]
        trainer.train_data_loader = data
        original = trainer._loss_parts
        measured = []

        def measure(output, labels):
            main, aux = original(output, labels)
            measured.append((main.item(), aux.item(), len(labels)))
            return main, aux

        with patch.object(trainer, '_loss_parts', side_effect=measure):
            trainer.training(1)
        self.assertEqual(trainer.train_epoch_stats['train_samples'], 5)
        for name, index in (('main', 0), ('aux', 1)):
            expected = sum(values[index] * values[2] for values in measured) / 5
            self.assertAlmostEqual(trainer.train_epoch_stats[f'train_loss_{name}'], expected)
        improved = trainer.validation(1)
        trainer.save_checkpoint(1, improved)
        row = self.rows(trainer)[0]
        self.assertAlmostEqual(row['train_loss_total'], row['train_loss_main'] + row['train_loss_aux'])
        self.assertEqual(row['optimizer_steps'], 2)
        self.assertEqual(row['optimizer_skipped_steps'], 0)
        self.assertEqual(row['lr_backbone'], 0)
        self.assertFalse(row['backbone_training'])
        self.assertIsNone(row['gpu_peak_memory_mb'])
        self.assertGreater(row['train_seconds'], 0)
        self.assertGreater(row['val_seconds'], 0)
        checkpoint = load_checkpoint(trainer.save_pth3 / 'last.pth')
        self.assertEqual(checkpoint['epoch_metrics'], row)
        self.assertEqual(checkpoint['best_epochs']['best_iou'], 1)
        summary = json.loads((trainer.save_folder3 / 'summary.json').read_text())
        self.assertEqual(summary['last_completed_epoch'], 1)
        self.assertEqual(summary['best']['best_iou']['epoch'], 1)
        self.assertTrue((trainer.save_folder3 / summary['best']['best_iou']['checkpoint']).exists())
        with (trainer.save_folder3 / 'history.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(csv_rows[0]['train_loss_total']), row['train_loss_total'])
        self.assertNotIn('\r', (trainer.save_folder3 / 'train.log').read_text())

    def test_teacher_total_keeps_auxiliary_loss_and_records_raw_kd(self):
        trainer = self.trainer('--loss', 'BCE')
        trainer.teacher = AuxiliaryModel().requires_grad_(False).eval()
        trainer.training(1)
        stats = trainer.train_epoch_stats
        self.assertGreater(stats['train_loss_aux'], 0)
        self.assertGreater(stats['train_loss_kd'], 0)
        expected = trainer.args.alpha * (stats['train_loss_main'] + stats['train_loss_aux'])
        expected += (1 - trainer.args.alpha) * stats['train_loss_kd']
        self.assertAlmostEqual(stats['train_loss_total'], expected, places=5)

    def test_actual_lr_records_pre_step_rates_and_backbone_unfreeze(self):
        trainer = self.trainer('--epoch_training_backbone', '1')
        rates = []
        original = trainer.optimizer.step

        def step(*args, **kwargs):
            rates.append([group['lr'] for group in trainer.optimizer.param_groups])
            return original(*args, **kwargs)

        with patch.object(trainer.optimizer, 'step', side_effect=step):
            trainer.training(1)
        for name, index in (('head', 0), ('backbone', 1)):
            self.assertEqual(trainer.train_epoch_stats[f'lr_{name}_first'], rates[0][index])
            self.assertEqual(trainer.train_epoch_stats[f'lr_{name}_last'], rates[-1][index])
            self.assertAlmostEqual(trainer.train_epoch_stats[f'lr_{name}'], np.mean(rates, axis=0)[index])
        self.assertTrue(trainer.train_epoch_stats['backbone_training'])

    def test_amp_skip_counter_does_not_advance_scheduler_or_report_unused_lr(self):
        trainer = self.trainer()

        class OverflowScaler:
            scale_value = 8.

            def is_enabled(self):
                return False

            def scale(self, loss):
                return loss

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                pass  # Simulate overflow for every batch.

            def update(self):
                self.scale_value /= 2

        trainer.scaler = OverflowScaler()
        before = trainer.lr_scheduler.state_dict()
        trainer.training(1)
        self.assertEqual(before, trainer.lr_scheduler.state_dict())
        self.assertEqual(trainer.train_epoch_stats['optimizer_skipped_steps'], 2)
        self.assertEqual(trainer.train_epoch_stats['optimizer_steps'], 0)
        self.assertEqual(trainer.train_epoch_stats['lr_head'], 0)
        self.assertEqual(trainer.train_epoch_stats['amp_scale'], 2)

    def test_validation_not_run_is_null_even_after_preceding_validation(self):
        trainer = self.trainer()
        self.epoch(trainer, 1)
        self.epoch(trainer, 2, validate=False)
        for name in (*VAL_FIELDS, 'val_samples', 'val_seconds'):
            self.assertIsNone(self.rows(trainer)[1][name], name)

    def test_periodic_checkpoint_default_boundary_keeps_epoch_without_validation(self):
        trainer = self.trainer('--epochs', '201', '--val_split', '0')
        self.assertEqual(trainer.args.save_freq, 100)
        self.epoch(trainer, 99, validate=False)
        archive = trainer.history.session_dir / 'checkpoint'
        self.assertFalse((archive / 'epoch_000099.pth').exists())
        self.epoch(trainer, 100, validate=False)
        periodic = archive / 'epoch_000100.pth'
        saved_bytes = periodic.read_bytes()
        saved = load_checkpoint(periodic)
        self.assertEqual(saved['epoch'], 100)
        self.assertIsNone(saved['epoch_metrics']['val_iou'])
        for key in ('model', 'optimizer', 'lr_scheduler', 'scaler', 'rng_state'):
            self.assertIn(key, saved)
        self.epoch(trainer, 101, validate=False)
        self.assertEqual(periodic.read_bytes(), saved_bytes)
        self.assertFalse((archive / 'epoch_000101.pth').exists())
        self.assertEqual(load_checkpoint(trainer.save_pth3 / 'last.pth')['epoch'], 101)
        self.assertEqual([row['epoch'] for row in self.rows(trainer)], [99, 100, 101])
        self.assertIn('Periodic checkpoint at epoch 100', (trainer.save_folder3 / 'train.log').read_text())
        resumed = self.trainer('--epochs', '201', '--val_split', '0', '--resume', str(periodic))
        self.assertEqual(resumed.args.start_epoch, 101)
        self.assertEqual(resumed.save_folder3, trainer.save_folder3.resolve())
        self.epoch(resumed, 200, validate=False)
        self.assertTrue((resumed.history.session_dir / 'checkpoint' / 'epoch_000200.pth').is_file())
        self.assertEqual(periodic.read_bytes(), saved_bytes)

    def test_periodic_checkpoint_with_no_improvement_and_best_on_same_epoch(self):
        trainer = self.trainer('--save_freq', '1')
        self.epoch(trainer, 1)
        archive = trainer.history.session_dir / 'checkpoint'
        self.assertEqual(list(archive.glob('*.pth')), [archive / 'epoch_000001.pth'])
        best_before = {name: (trainer.save_folder3 / path).read_bytes()
                       for name, path in trainer.best_checkpoints.items() if path}
        trainer.training(2)
        # Restore the preceding model so validation repeats its metrics exactly.
        trainer.model.load_state_dict(load_checkpoint(archive / 'epoch_000001.pth')['model'])
        improved = trainer.validation(2)
        self.assertEqual(improved, [])
        trainer.save_checkpoint(2, improved)
        self.assertTrue((archive / 'epoch_000002.pth').exists())
        self.assertIsNotNone(load_checkpoint(archive / 'epoch_000002.pth')['epoch_metrics']['val_iou'])
        for name, expected in best_before.items():
            self.assertEqual((trainer.save_pth3 / f'{name}.pth').read_bytes(), expected)

    def test_periodic_checkpoint_disabled_still_saves_best_and_logs(self):
        trainer = self.trainer('--epochs', '101', '--save_freq', '0')
        self.epoch(trainer, 1)
        self.assertTrue((trainer.save_pth3 / 'best_iou.pth').is_file())
        self.epoch(trainer, 100, validate=False)
        self.assertFalse((trainer.history.session_dir / 'checkpoint' / 'epoch_000100.pth').exists())
        self.assertEqual(load_checkpoint(trainer.save_pth3 / 'last.pth')['epoch'], 100)
        self.assertEqual(len(self.rows(trainer)), 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--save_freq', '-1')

    def test_run_loop_no_validation_and_eval_frequency(self):
        for options, expected in [(('--val_split', '0'), [False, False]),
                                   (('--eval_freq', '2'), [False, True])]:
            trainer = run_training(self.args('--epochs', '2', *options))
            rows = self.rows(trainer)
            self.assertEqual([row['val_loss'] is not None for row in rows], expected)
            self.assertEqual(json.loads((trainer.save_folder3 / 'summary.json').read_text())['status'], 'completed')

    def test_resume_old_epoch_appends_session_and_keeps_old_best_snapshot(self):
        trainer = self.trainer()
        self.epoch(trainer, 1)
        old_checkpoint = trainer.save_pth3 / 'saved_epoch1.pth'
        old_checkpoint.write_bytes((trainer.save_pth3 / 'last.pth').read_bytes())
        old_best_path = trainer.save_folder3 / trainer.best_checkpoints['best_iou']
        old_best_bytes = old_best_path.read_bytes()
        self.epoch(trainer, 2)
        resumed = self.trainer('--resume', str(old_checkpoint))
        self.assertNotEqual(trainer.session_id, resumed.session_id)
        self.assertEqual(trainer.tb_image_samples, resumed.tb_image_samples)
        self.assertEqual(old_best_path.read_bytes(), old_best_bytes)
        self.epoch(resumed, 2)
        rows = self.rows(resumed)
        self.assertEqual([row['epoch'] for row in rows], [1, 2, 2])
        self.assertNotEqual(rows[1]['session_id'], rows[2]['session_id'])
        snapshots = list((trainer.save_folder3 / 'sessions').glob('*/session.json'))
        self.assertEqual(len(snapshots), 2)
        snapshot = json.loads((resumed.history.session_dir / 'session.json').read_text())
        self.assertEqual(snapshot['resumed_checkpoint_epoch'], 1)
        self.assertEqual(snapshot['pytorch'], str(torch.__version__))
        self.assertEqual(snapshot['config']['loss_version'], CUSTOM_LOSS_VERSION)
        # Archived checkpoints can also be used as the resume input.
        archived_resume = self.trainer('--resume', str(old_best_path))
        self.assertEqual(archived_resume.save_folder3, trainer.save_folder3.resolve())

    def test_legacy_checkpoint_gets_versioned_metadata_without_inventing_history(self):
        trainer = self.trainer()
        self.epoch(trainer, 1)
        checkpoint = load_checkpoint(trainer.save_pth3 / 'last.pth')
        for key in ('best_epochs', 'best_checkpoints', 'epoch_metrics', 'loss_version',
                    'tb_image_samples', 'session_id'):
            checkpoint.pop(key)
        checkpoint['args'].pop('loss_version')
        path = trainer.save_pth3 / 'legacy.pth'
        torch.save(checkpoint, path)
        resumed = self.trainer('--resume', str(path))
        self.assertEqual(resumed.best_epochs['best_iou'], 1)
        self.assertEqual(len(self.rows(resumed)), 1)
        self.epoch(resumed, 2)
        self.assertEqual(load_checkpoint(resumed.save_pth3 / 'last.pth')['loss_version'], CUSTOM_LOSS_VERSION)
        self.assertEqual(len(self.rows(resumed)), 2)

    def test_interrupt_or_error_preserves_completed_epochs_and_traceback(self):
        original = Trainer.training
        for exception, status in ((RuntimeError('intentional failure'), 'failed'),
                                  (KeyboardInterrupt(), 'interrupted')):
            def fail_second(trainer, epoch):
                if epoch == 2:
                    raise exception
                return original(trainer, epoch)

            before = set((self.root / 'runs').glob('*'))
            with patch.object(Trainer, 'training', fail_second), self.assertRaises(type(exception)):
                run_training(self.args())
            folder = (set((self.root / 'runs').glob('*')) - before).pop()
            summary = json.loads((folder / 'summary.json').read_text())
            self.assertEqual(summary['status'], status)
            self.assertEqual(summary['last_completed_epoch'], 1)
            self.assertEqual(len((folder / 'history.jsonl').read_text().splitlines()), 1)
            self.assertIn('Traceback', (folder / 'train.log').read_text())

    def test_startup_error_is_logged(self):
        with patch('tool.train.ModelBuilder', side_effect=RuntimeError('model startup error')):
            with self.assertRaisesRegex(RuntimeError, 'startup error'):
                run_training(self.args())
        folder = next((self.root / 'runs').glob('*'))
        self.assertEqual(json.loads((folder / 'summary.json').read_text())['status'], 'failed')
        self.assertIn('model startup error', (folder / 'train.log').read_text())

    def test_tb_first_periodic_best_only_once_and_no_extra_forward(self):
        trainer = self.trainer('--tb_image_freq', '3')
        writer = trainer.writer = RecordingWriter()
        with patch.object(trainer.model, 'forward', wraps=trainer.model.forward) as forward:
            trainer.validation(1)  # First + best: only one set.
            self.assertEqual(forward.call_count, len(trainer.val_data_loader))
            self.assertEqual(len(writer.images), len(trainer.tb_sample_indices))
            trainer.validation(2)  # Identical weights, no best or periodic trigger.
            self.assertEqual(len(writer.images), len(trainer.tb_sample_indices))
            trainer.validation(3)  # Periodic.
            trainer.validation(3)  # A second call at the same epoch is deduplicated.
            self.assertEqual(len(writer.images), len(trainer.tb_sample_indices) * 2)
            trainer.best_iou = -1
            trainer.validation(4)  # New best outside periodic schedule.
            self.assertEqual(len(writer.images), len(trainer.tb_sample_indices) * 3)
            self.assertEqual(forward.call_count, len(trainer.val_data_loader) * 5)
        trainer.training(4)
        trainer.save_checkpoint(4)
        self.assertTrue(any(tag == 'Epoch/train_loss_total' for tag, _, _ in writer.scalars))
        self.assertFalse(list(trainer.save_folder3.rglob('*.png')))
        self.assertFalse(list(trainer.save_folder3.rglob('*.jpg')))
        panel = writer.images[0][1]
        self.assertEqual(panel.shape, (3, 32, 128))
        self.assertGreaterEqual(panel.min().item(), 0)
        self.assertLessEqual(panel.max().item(), 1)

    def test_real_tensorboard_event_files_when_installed(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest('TensorBoard is optional and not installed in this environment')
        trainer = self.trainer('--tensorboard')
        self.epoch(trainer, 1)
        trainer.writer.flush()
        event_dir = trainer.save_folder3 / 'tensorboard' / trainer.session_id
        events = EventAccumulator(str(event_dir)).Reload()
        self.assertIn('Epoch/train_loss_total', events.Tags()['scalars'])
        self.assertEqual(len(events.Tags()['images']), len(trainer.tb_sample_indices))
        self.assertFalse(list(trainer.save_folder3.rglob('*.png')))
        resumed = self.trainer('--resume', str(trainer.save_pth3 / 'last.pth'), '--tensorboard')
        self.epoch(resumed, 2)
        self.assertTrue((trainer.save_folder3 / 'tensorboard' / resumed.session_id).is_dir())
        self.assertEqual(trainer.tb_image_samples, resumed.tb_image_samples)


class PreviewChecks(unittest.TestCase):
    def test_selection_prioritizes_three_positive_one_empty_and_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sequence in ('a', 'b', 'c', 'd'):
                _, masks = make_sequence(root, count=4, sequence=sequence)
                Image.new('L', (32, 32)).save(masks / 'frame_3.bmp')
            dataset = FrameMaskDataset(root / 'images', root / 'masks', img_size=(32, 32))
            indices, saved = select_preview_samples(dataset, 4)
            self.assertEqual(sum(bool(dataset[index][2].any()) for index in indices), 3)
            self.assertEqual(len({dataset.samples[index][0].parent for index in indices}), 4)
            resumed, identities = select_preview_samples(dataset, 4, saved)
            self.assertEqual((indices, saved), (resumed, identities))
            self.assertEqual(len(select_preview_samples(dataset, 20)[0]), len(dataset))

    def test_empty_or_positive_category_falls_back_and_panel_uses_current_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, masks = make_sequence(root, count=6)
            dataset = FrameMaskDataset(root / 'images', root / 'masks')
            self.assertEqual(len(select_preview_samples(dataset, 4)[0]), 4)
            for path in masks.glob('*'):
                Image.new('L', (32, 32)).save(path)
            self.assertEqual(len(select_preview_samples(dataset, 4)[0]), 4)
            current, _, labels = dataset[0]
            panel = preview_panel(current, labels, torch.zeros_like(labels))
            torch.testing.assert_close(panel[:, :, :32], current / 255)
            torch.testing.assert_close(panel[:, :, -32:], torch.full((3, 32, 32), 0.5))


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    with redirect_stdout(io.StringIO()):
        unittest.main(verbosity=2)
