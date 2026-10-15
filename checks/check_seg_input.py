"""Behavior checks for configurable Flow/RGB heads and their export/benchmark path."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from NeuFlow import config
from NeuFlow.model_builder import ModelBuilder
from external.ILNet.model.LW_Opticalflow import SEBlock
from tool.checkpoints import (load_checkpoint, load_model_weights, resolve_seg_input,
                              transfer_model_weights)
from tool.export_seg import ExportModel, compare_logits, export_model, validation_pairs
from tool.benchmark_seg import compare_timings, cost_estimate, make_model_pair, stats
from tool.infer import parse_args as infer_args, resolve_config, run_inference
from tool.train import Trainer, build_datasets, parse_args as train_args
from checks.check_frame_training import make_sequence, QuietProgress


def checkpoint_of(model):
    return {'model': model.state_dict(), 'args': {'seg_input': model.seg_input}}


class ConfigurationChecks(unittest.TestCase):
    def test_defaults_and_legacy_checkpoint_ignore_changed_global_config(self):
        with patch.object(config, 'seg_input', 'flow_rgb'):
            self.assertEqual(ModelBuilder().seg_input, 'flow_rgb')
            self.assertEqual(resolve_seg_input(), 'flow_rgb')
            self.assertEqual(resolve_seg_input(checkpoint={}), 'flow')
            self.assertEqual(ModelBuilder('flow').seg_input, 'flow')
        with self.assertRaisesRegex(ValueError, 'seg_input'):
            ModelBuilder('rgb')

    def test_checkpoint_conflict_and_flow_only_dataset_rejected_early(self):
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_seg_input('flow_rgb', {})
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_seg_input('flow', {'args': {'seg_input': 'flow_rgb'}})
        args = train_args(['--dataset', 'opticalflow', '--data_path', '/not/needed', '--seg_input', 'flow_rgb'])
        with self.assertRaisesRegex(ValueError, 'requires --dataset frames'):
            build_datasets(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_args(['--data_path', '/not/needed', '--resume', 'a.pth', '--transfer', 'b.pth'])

    def test_inference_restores_architecture_and_rejects_override(self):
        command = ['--checkpoint', 'm.pth', '--current', 'a.png', '--past', 'b.png', '--output', 'out.png']
        saved = {'args': {'seg_input': 'flow_rgb', 'img_size': [32, 32]}}
        self.assertEqual(resolve_config(infer_args(command), saved).seg_input, 'flow_rgb')
        self.assertEqual(resolve_config(infer_args(command), {}).seg_input, 'flow')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_config(infer_args(command + ['--seg_input', 'flow']), saved)


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_channel_order_normalization_and_pool_flow_only(self):
        model = ModelBuilder('flow_rgb').eval()
        current = torch.rand(1, 3, 32, 32) * 255
        past = torch.rand_like(current) * 255
        before_current, before_past = current.clone(), past.clone()
        captured = {}
        def flow_hook(module, inputs, output):
            captured['flow'] = output[-1].detach().clone()
        def head_hook(module, inputs):
            captured['head'] = inputs[0].detach().clone()
        def pool_hook(module, inputs):
            captured['pool'] = inputs[0].detach().clone()
        handles = [model.backbone_op.register_forward_hook(flow_hook),
                   model.seg_head.register_forward_pre_hook(head_hook),
                   model.seg_head.initial_block.ext_branch.register_forward_pre_hook(pool_hook)]
        try:
            with torch.inference_mode():
                output = model(current, past)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.shape, (1, 1, 32, 32))
        torch.testing.assert_close(captured['head'][:, :2], model.flow_normalize_seg(captured['flow']))
        torch.testing.assert_close(captured['head'][:, 2:], current / 255, rtol=0, atol=0)
        torch.testing.assert_close(captured['pool'], captured['head'][:, :2], rtol=0, atol=0)
        torch.testing.assert_close(current, before_current, rtol=0, atol=0)
        torch.testing.assert_close(past, before_past, rtol=0, atol=0)
        self.assertEqual(tuple(model.seg_head.initial_block.main_branch.weight.shape), (6, 5, 3, 3))

    def test_transfer_is_equivalent_and_new_rgb_weights_receive_gradients(self):
        baseline, rgb = ModelBuilder('flow').eval(), ModelBuilder('flow_rgb').eval()
        # Legacy checkpoints can have no architecture metadata and module. prefixes.
        legacy = {'state_dict': {'module.' + key: value for key, value in baseline.state_dict().items()}}
        transfer_model_weights(rgb, legacy)
        old_weight = baseline.seg_head.initial_block.main_branch.weight
        new_weight = rgb.seg_head.initial_block.main_branch.weight
        torch.testing.assert_close(new_weight[:, :2], old_weight, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(new_weight[:, 2:]).item(), 0)
        current, past = validation_pairs(32, 32)[0]
        with torch.inference_mode():
            for iterations in (8, 10):
                torch.testing.assert_close(baseline(current, past, iters_s8=iterations),
                                           rgb(current, past, iters_s8=iterations), rtol=1e-5, atol=1e-6)
        prediction = rgb(current, past, iters_s8=2)
        F.binary_cross_entropy_with_logits(prediction, torch.ones_like(prediction)).backward()
        gradient = new_weight.grad[:, 2:]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        # Once the RGB weights learn, RGB must actually affect head output for fixed flow.
        with torch.no_grad():
            new_weight[:, 2:].fill_(0.1)
            fixed_flow = torch.randn(1, 2, 32, 32)
            a = rgb.seg_head(torch.cat((fixed_flow, torch.zeros(1, 3, 32, 32)), dim=1))
            b = rgb.seg_head(torch.cat((fixed_flow, torch.ones(1, 3, 32, 32)), dim=1))
            self.assertGreater((a - b).abs().max().item(), 0)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            transfer_model_weights(baseline, checkpoint_of(rgb))
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            load_model_weights(rgb, legacy)

    def test_backward_reaches_all_refinements_in_both_modes(self):
        for mode in config.SEG_INPUT_MODES:
            with self.subTest(mode=mode):
                model = ModelBuilder(mode).train()
                deltas = []
                def retain(module, inputs, output):
                    output[1].retain_grad()
                    deltas.append(output[1])
                handles = [model.backbone_op.refine_s16.register_forward_hook(retain),
                           model.backbone_op.refine_s8.register_forward_hook(retain)]
                current, past = torch.rand(2, 3, 32, 32) * 255, torch.rand(2, 3, 32, 32) * 255
                outputs = model(current, past, iters_s16=2, iters_s8=3)
                loss = sum(F.binary_cross_entropy_with_logits(out, torch.ones_like(out)) for out in outputs)
                loss.backward()
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(deltas), 5)
                for delta in deltas:
                    self.assertIsNotNone(delta.grad)
                    self.assertTrue(torch.isfinite(delta.grad).all())
                    self.assertGreater(torch.count_nonzero(delta.grad).item(), 0)

    def test_export_copy_folds_empty_se_without_changing_model(self):
        current, past = validation_pairs(32, 32)[0]
        for mode in config.SEG_INPUT_MODES:
            with self.subTest(mode=mode):
                model = ModelBuilder(mode).eval()
                empty = lambda module: isinstance(module, SEBlock) and module.fc[0].out_features == 0
                original_count = sum(empty(m) for m in model.modules())
                wrapper = ExportModel(model, 32, 32)
                self.assertGreater(original_count, 0)
                self.assertEqual(wrapper.folded_se_blocks, original_count)
                self.assertEqual(sum(empty(m) for m in model.modules()), original_count)
                self.assertEqual(sum(empty(m) for m in wrapper.modules()), 0)
                with torch.inference_mode():
                    torch.testing.assert_close(wrapper(current, past), model(current, past), rtol=0, atol=0)

    def test_export_trace_handles_both_input_modes_and_iteration_counts(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        current, past = validation_pairs(32, 32)[0]
        models = make_model_pair(seed=42)
        for mode, model in models.items():
            for iterations in (8, 10):
                with self.subTest(mode=mode, iters_s8=iterations):
                    wrapper = ExportModel(model, 32, 32, iters_s8=iterations)
                    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                        traced = torch.jit.trace(wrapper, (current, past), check_trace=True)
                        torch.testing.assert_close(traced(current, past), wrapper(current, past))


class TrainingChecks(unittest.TestCase):
    @patch('tool.train.tqdm', QuietProgress)
    def test_rgb_train_resume_infer_and_transfer_with_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, _ = make_sequence(root, count=3)
            base = ['--data_path', str(root), '--val_split', '0', '--img_size', '32', '32',
                    '--batch_size', '2', '--num_workers', '0', '--device', 'cpu', '--no-amp',
                    '--no-tensorboard', '--epochs', '3', '--warm_up_epochs', '0', '--loss', 'BCE',
                    '--backbone_op_pretrained', '', '--work_dir', str(root / 'runs'), '--stabilization', 'none']
            original = Trainer(train_args(base + ['--seg_input', 'flow']))
            original.training(1)
            original.save_checkpoint(1)
            old_path = original.save_pth3 / 'last.pth'
            converted = Trainer(train_args(base + ['--seg_input', 'flow_rgb', '--transfer', str(old_path)]))
            self.assertEqual(converted.args.start_epoch, 1)
            self.assertEqual(len(converted.optimizer.state), 0)
            self.assertEqual(converted.model.seg_input, 'flow_rgb')
            converted.training(1)
            converted.save_checkpoint(1)
            path = converted.save_pth3 / 'last.pth'
            self.assertEqual(load_checkpoint(path)['args']['seg_input'], 'flow_rgb')
            self.assertEqual(json.loads((converted.save_folder3 / 'config.json').read_text())['seg_input'], 'flow_rgb')
            resumed = Trainer(train_args(base + ['--resume', str(path)]))
            self.assertEqual(resumed.model.seg_input, 'flow_rgb')
            self.assertEqual(resumed.args.start_epoch, 2)
            for name, expected in converted.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)
            torch.manual_seed(99)
            converted.training(2)
            torch.manual_seed(99)
            resumed.training(2)
            for name, expected in converted.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                Trainer(train_args(base + ['--resume', str(path), '--seg_input', 'flow']))
            for iterations in (8, 10):
                args = infer_args(['--checkpoint', str(path), '--current', str(images / 'frame_2.png'),
                                   '--past', str(images / 'frame_1.png'), '--output', str(root / f'mask_{iterations}.png'),
                                   '--device', 'cpu', '--no-amp', '--stabilization', 'none', '--iters_s8', str(iterations)])
                with patch('data_utils.frame_pair.stabilize_past', side_effect=AssertionError('Must not warp again')):
                    mask, alignment = run_inference(args)
                self.assertEqual(mask.shape, (32, 32))
                self.assertEqual(alignment.reason, 'disabled')


class BenchmarkChecks(unittest.TestCase):
    def test_pair_has_identical_common_weights_and_nonzero_rgb(self):
        models = make_model_pair(seed=17)
        old, new = models['flow'].state_dict(), models['flow_rgb'].state_dict()
        key = 'seg_head.initial_block.main_branch.weight'
        for name, expected in old.items():
            torch.testing.assert_close(new[name][:, :2] if name == key else new[name], expected, rtol=0, atol=0)
        self.assertGreater(torch.count_nonzero(new[key][:, 2:]).item(), 0)

    def test_estimates_and_latency_decisions_do_not_confuse_macs_with_time(self):
        estimate = cost_estimate((960, 544))
        self.assertEqual(estimate['extra_parameters'], 162)
        self.assertEqual(estimate['extra_conv_macs'], 21150720)
        self.assertIsNone(estimate['latency_ms'])
        baseline = stats([10.] * 99 + [100.])
        self.assertEqual(baseline['median_ms'], 10.)
        self.assertTrue(compare_timings(baseline, stats([10.4] * 100))['within_limit'])
        # Median passes, p95 fails: must not accept the model.
        self.assertFalse(compare_timings(baseline, stats([10.4] * 90 + [12.] * 10))['within_limit'])
        with self.assertRaises(ValueError):
            stats([float('nan')])

    def test_logit_validation_detects_shape_nonfinite_and_numerical_errors(self):
        a = np.ones((1, 1, 4, 4), dtype=np.float32)
        self.assertTrue(compare_logits(a, a)['passed'])
        self.assertFalse(compare_logits(a, a * 2)['passed'])
        for invalid in (a.reshape(1, 16), a * float('nan')):
            with self.assertRaises(ValueError):
                compare_logits(a, invalid)

    def test_unavailable_gpu_never_produces_a_latency_pass(self):
        from tool.benchmark_seg import main
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(SystemExit) as error:
                main(['--phase', 'run', '--output_dir', folder])
            self.assertEqual(error.exception.code, 2)
            report = json.loads((Path(folder) / 'report.json').read_text())
            self.assertEqual(report['status'], 'not_measured')
            self.assertIsNone(report['within_five_percent'])


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                     'Install tool/requirements-export.txt to verify ONNX')
class OnnxChecks(unittest.TestCase):
    def test_full_models_export_and_match_pytorch_at_eight_and_ten_iterations(self):
        models = make_model_pair(seed=42)
        with tempfile.TemporaryDirectory() as folder:
            for mode, model in models.items():
                for iterations in (8, 10):
                    with self.subTest(mode=mode, iters_s8=iterations):
                        path = Path(folder) / f'{mode}_{iterations}.onnx'
                        metadata = export_model(model, path, (32, 32), iters_s8=iterations)
                        self.assertTrue(all(item['passed'] for item in metadata['validation']))
                        self.assertEqual(metadata['seg_input'], mode)
                        self.assertTrue(path.with_suffix('.validation.npz').is_file())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
