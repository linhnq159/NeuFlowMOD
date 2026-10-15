"""Prepare matched Flow/Flow+RGB ONNX models and benchmark TensorRT FP16 on a GPU.

Use --phase estimate without ONNX/TensorRT, prepare on a CPU workstation, and
run on the target GPU. Results from a desktop GPU are not estimates of Orin latency.
"""

import csv
import gc
import json
import platform
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from NeuFlow.model_builder import ModelBuilder
from tool.checkpoints import load_checkpoint, load_model_weights, load_weights, resolve_seg_input, transfer_model_weights
from tool.export_seg import compare_logits, export_model, sha256_file, validate_shape_iterations


def cost_estimate(img_size):
    width, height = img_size
    pixels = width * height
    return dict(kind='arithmetic estimate, not measured latency', img_size=list(img_size),
                extra_parameters=6 * 3 * 3 * 3,
                extra_conv_macs=pixels // 4 * 6 * 3 * 3 * 3,
                extra_rgb_normalization_elements=3 * pixels,
                unfused_fp16_rgb_bytes=3 * pixels * 2,
                unfused_fp16_concat_bytes=5 * pixels * 2,
                note='TensorRT may fuse buffers/operations. These sizes are not peak runtime memory.',
                latency_ms=None, within_five_percent=None)


def make_model_pair(checkpoint=None, backbone_pretrained=None, seed=1211):
    torch.manual_seed(seed)
    flow = ModelBuilder(seg_input='flow').eval()
    if checkpoint is not None:
        resolve_seg_input('flow', checkpoint)
        load_model_weights(flow, checkpoint)
    elif backbone_pretrained:
        load_weights(flow.backbone_op, backbone_pretrained)
    rgb = ModelBuilder(seg_input='flow_rgb').eval()
    transfer_model_weights(rgb, {'model': flow.state_dict(), 'args': {'seg_input': 'flow'}})
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Checkpoint conversion uses zeros; timing must exercise a nonzero RGB path.
        rgb.seg_head.initial_block.main_branch.weight[:, 2:].normal_(0, 0.01, generator=generator)
    return {'flow': flow, 'flow_rgb': rgb}


def prepare(args):
    folder = Path(args.output_dir)
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    models = make_model_pair(checkpoint, args.backbone_pretrained if checkpoint is None else None, args.seed)
    artifacts = []
    for iterations in args.iters_s8:
        for mode, model in models.items():
            output = folder / f'{mode}_s8_{iterations}.onnx'
            metadata = export_model(model, output, args.img_size, args.iters_s16, iterations, args.seed)
            artifacts.append(dict(mode=mode, iters_s8=iterations, onnx=output.name,
                                  metadata=output.with_suffix('.json').name,
                                  sha256=metadata['onnx_sha256']))
    manifest = dict(version=1, img_size=args.img_size, batch_size=1, iters_s16=args.iters_s16,
                    iters_s8=args.iters_s8, seed=args.seed, models=artifacts,
                    source_checkpoint=args.checkpoint, backbone_pretrained=args.backbone_pretrained,
                    rgb_weights='copied flow weights + seeded nonzero RGB weights; latency experiment only',
                    estimate=cost_estimate(args.img_size))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def device_snapshot():
    result = {}
    # Read-only diagnostics; never change power/clock settings on the user's device.
    for label, command in [
        ('nvidia_smi', ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu', '--format=csv,noheader']),
        ('jetson_power_mode', ['nvpmodel', '-q']),
    ]:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=3)
            if proc.returncode == 0:
                result[label] = proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    release = Path('/etc/nv_tegra_release')
    if release.exists():
        result['jetpack_release'] = release.read_text().strip()
    temperatures = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000
        except (OSError, ValueError):
            pass
    result['temperatures_c'] = temperatures
    return result


def build_engine(trt, onnx_path, workspace_mib):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'TensorRT could not parse {onnx_path}:\n{errors}')
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT engine build failed: {onnx_path}')
    engine_path = Path(onnx_path).with_suffix('.engine')
    engine_path.write_bytes(bytes(serialized))
    # Always rebuild on the measuring GPU. Serialized engines are not portable to Orin.
    return engine_path


class TensorRTRunner:
    """Static batch-one TensorRT >= 8.6 runner, with preallocated CUDA/pinned buffers."""

    def __init__(self, trt, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Unable to deserialize {engine_path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Unable to allocate TensorRT execution context.')
        self.stream = torch.cuda.Stream()
        self.device, self.host = {}, {}
        expected = {'current': trt.TensorIOMode.INPUT, 'past': trt.TensorIOMode.INPUT,
                    'logits': trt.TensorIOMode.OUTPUT}
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != set(expected):
            raise ValueError(f'Unexpected engine I/O: {names}')
        for name, mode in expected.items():
            if self.engine.get_tensor_mode(name) != mode or self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f'Expected FP32 {mode} binding for {name}; internal compute uses FP16.')
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(n < 1 for n in shape):
                raise ValueError('Benchmark requires fixed positive input/output shapes.')
            self.device[name] = torch.empty(shape, device='cuda', dtype=torch.float32)
            self.host[name] = torch.empty(shape, pin_memory=True, dtype=torch.float32)
            if not self.context.set_tensor_address(name, self.device[name].data_ptr()):
                raise RuntimeError(f'Unable to bind {name}')
        self.context_memory_bytes = int(self.engine.device_memory_size)
        self.start_event, self.end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def enqueue(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed.')

    def load_inputs(self, current, past):
        for name, array in [('current', current), ('past', past)]:
            self.host[name].copy_(torch.from_numpy(array))
        with torch.cuda.stream(self.stream):
            for name in ('current', 'past'):
                self.device[name].copy_(self.host[name], non_blocking=True)
        self.stream.synchronize()

    def logits(self):
        with torch.cuda.stream(self.stream):
            self.enqueue()
            self.host['logits'].copy_(self.device['logits'], non_blocking=True)
        self.stream.synchronize()
        return self.host['logits'].numpy().copy()

    def sample(self, transfers):
        start = time.perf_counter()
        with torch.cuda.stream(self.stream):
            if transfers:
                for name in ('current', 'past'):
                    self.device[name].copy_(self.host[name], non_blocking=True)
            self.start_event.record(self.stream)
            self.enqueue()
            self.end_event.record(self.stream)
            if transfers:
                self.host['logits'].copy_(self.device['logits'], non_blocking=True)
        self.stream.synchronize()
        host_ms = (time.perf_counter() - start) * 1000
        return float(self.start_event.elapsed_time(self.end_event)), host_ms

    def close(self):
        self.stream.synchronize()
        self.context = None
        self.device.clear()
        self.host.clear()
        self.engine = None
        self.runtime = None
        gc.collect()
        torch.cuda.empty_cache()


def stats(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all() or np.any(samples <= 0):
        raise ValueError('Timing samples must be finite and positive.')
    return dict(count=int(samples.size), median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples, 95)), mean_ms=float(samples.mean()))


def compare_timings(baseline, rgb, limit_percent=5.0):
    result = {}
    for metric in ('median_ms', 'p95_ms'):
        delta = rgb[metric] - baseline[metric]
        result[metric] = dict(delta_ms=delta, increase_percent=100 * delta / baseline[metric])
    result['within_limit'] = all(result[key]['increase_percent'] <= limit_percent
                                 for key in ('median_ms', 'p95_ms'))
    return result


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. No GPU latency measurement was performed.')
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError('TensorRT >= 8.6 is required on the measuring GPU; use JetPack on Jetson.') from error
    version = tuple(int(part) for part in trt.__version__.split('.')[:2])
    if version < (8, 6):
        raise RuntimeError('TensorRT >= 8.6 is required for the named tensor I/O API.')
    folder = Path(args.output_dir)
    manifest = json.loads((folder / 'manifest.json').read_text())
    entries = {}
    for artifact in manifest['models']:
        path = folder / artifact['onnx']
        metadata = json.loads((folder / artifact['metadata']).read_text())
        if sha256_file(path) != artifact['sha256'] or metadata['onnx_sha256'] != artifact['sha256']:
            raise ValueError(f'ONNX digest mismatch: {path}')
        if sha256_file(folder / metadata['validation_file']) != metadata['validation_sha256']:
            raise ValueError(f'Validation data digest mismatch: {path}')
        if not metadata['validation'] or not all(check['passed'] for check in metadata['validation']):
            raise ValueError(f'ONNX validation missing or failed: {path}')
        if (metadata['img_size'] != manifest['img_size'] or metadata['batch_size'] != 1
                or metadata['iters_s16'] != manifest['iters_s16']
                or metadata['iters_s8'] != artifact['iters_s8'] or metadata['seg_input'] != artifact['mode']):
            raise ValueError(f'Inconsistent benchmark configuration: {path}')
        entries[(artifact['iters_s8'], artifact['mode'])] = (path, metadata)
    for iterations in manifest['iters_s8']:
        for mode in ('flow', 'flow_rgb'):
            if (iterations, mode) not in entries:
                raise ValueError(f'Missing {mode} model for S8={iterations}.')
    engines = {key: build_engine(trt, path, args.workspace_mib) for key, (path, _) in entries.items()}
    report = dict(status='running', gpu=torch.cuda.get_device_name(), platform=platform.platform(),
                  torch_version=torch.__version__, cuda_build=torch.version.cuda, tensorrt_version=trt.__version__,
                  precision='FP16 enabled, TF32 disabled; FP32 I/O and supported fallback layers',
                  cuda_graph=False, concurrency=1, workspace_mib=args.workspace_mib,
                  warmup_seconds=args.warmup_seconds, rounds=args.rounds, samples_per_round=args.samples,
                  latency_limit_percent=args.max_increase_percent, manifest=manifest,
                  reference_only_for_other_hardware=True, before=device_snapshot(), results=[], comparisons=[])
    rows = []
    for iterations in manifest['iters_s8']:
        for round_index in range(args.rounds):
            order = ('flow', 'flow_rgb') if round_index % 2 == 0 else ('flow_rgb', 'flow')
            for mode in order:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_before, total_memory = torch.cuda.mem_get_info()
                torch.cuda.reset_peak_memory_stats()
                runner = TensorRTRunner(trt, engines[(iterations, mode)])
                try:
                    metadata = entries[(iterations, mode)][1]
                    checks = []
                    with np.load(folder / metadata['validation_file']) as validation:
                        for index in range(2):
                            runner.load_inputs(validation[f'current_{index}'], validation[f'past_{index}'])
                            check = compare_logits(validation[f'logits_{index}'], runner.logits(),
                                                   atol=args.atol, rtol=args.rtol)
                            check['passed'] = check['passed'] and check['mask_agreement'] >= args.min_mask_agreement
                            checks.append(check)
                            if not check['passed']:
                                raise RuntimeError(f'TensorRT logit validation failed for {mode}: {check}')
                        runner.load_inputs(validation['current_0'], validation['past_0'])
                    observed_used = total_memory - torch.cuda.mem_get_info()[0]
                    result = dict(mode=mode, iters_s8=iterations, round=round_index, validation=checks,
                                  device_snapshot=device_snapshot(), timings={})
                    for transfers in (False, True):
                        label = 'with_transfers' if transfers else 'device_resident'
                        deadline = time.perf_counter() + args.warmup_seconds
                        while time.perf_counter() < deadline:
                            runner.sample(transfers)
                        gpu_times, host_times = [], []
                        for sample_index in range(args.samples):
                            gpu_ms, host_ms = runner.sample(transfers)
                            gpu_times.append(gpu_ms)
                            host_times.append(host_ms)
                            rows.append(dict(iters_s8=iterations, round=round_index, mode=mode,
                                             stage=label, sample=sample_index, gpu_ms=gpu_ms, host_ms=host_ms))
                        result['timings'][label] = dict(gpu=stats(gpu_times), host=stats(host_times))
                        observed_used = max(observed_used, total_memory - torch.cuda.mem_get_info()[0])
                    result['memory'] = dict(
                        trt_context_bytes=runner.context_memory_bytes,
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        device_used_peak_observed_bytes=observed_used,
                        device_used_before_load_bytes=total_memory - free_before,
                        note='Context memory is reported by TRT; PyTorch peak excludes TRT allocations. '
                             'Global device use is sampled after load and stages, not a continuous allocation trace.')
                    report['results'].append(result)
                    print(f'S8={iterations} round={round_index + 1} {mode}: '
                          f'{result["timings"]["device_resident"]["gpu"]}', flush=True)
                finally:
                    runner.close()
        paired = [item for item in report['results'] if item['iters_s8'] == iterations]
        comparisons = {}
        for label, metric in [('device_resident', 'gpu'), ('with_transfers', 'host')]:
            aggregate = {}
            for mode in ('flow', 'flow_rgb'):
                aggregate[mode] = stats([row[f'{metric}_ms'] for row in rows
                                         if row['iters_s8'] == iterations and row['stage'] == label and row['mode'] == mode])
            comparison = compare_timings(aggregate['flow'], aggregate['flow_rgb'], args.max_increase_percent)
            round_checks = []
            for round_index in range(args.rounds):
                by_mode = {item['mode']: item['timings'][label][metric]
                           for item in paired if item['round'] == round_index}
                round_checks.append(compare_timings(by_mode['flow'], by_mode['flow_rgb'], args.max_increase_percent))
            comparison.update(aggregate=aggregate, rounds=round_checks,
                              status=('failed' if not comparison['within_limit'] else
                                      'passed' if all(c['within_limit'] for c in round_checks) else 'inconclusive'))
            comparisons[label] = comparison
        report['comparisons'].append(dict(iters_s8=iterations, **comparisons))
    statuses = [item[label]['status'] for item in report['comparisons']
                for label in ('device_resident', 'with_transfers')]
    report['status'] = ('failed' if 'failed' in statuses else 'inconclusive' if 'inconclusive' in statuses else 'passed')
    report['after'] = device_snapshot()
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with (folder / 'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['iters_s8', 'round', 'mode', 'stage', 'sample', 'gpu_ms', 'host_ms'])
        writer.writeheader()
        writer.writerows(rows)
    return report


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['estimate', 'prepare', 'run', 'all'], default='all')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint', help='Optional trained flow-only ModelBuilder checkpoint')
    parser.add_argument('--backbone_pretrained', default=str(ROOT / 'neuflow_sintel.pth'))
    parser.add_argument('--img_size', nargs=2, type=int, default=[960, 544])
    parser.add_argument('--iters_s16', type=int, default=1)
    parser.add_argument('--iters_s8', nargs='+', type=int, default=[8, 10])
    parser.add_argument('--seed', type=int, default=1211)
    parser.add_argument('--cpu_threads', type=int, default=2)
    parser.add_argument('--warmup_seconds', type=float, default=3.0)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--workspace_mib', type=int, default=2048)
    parser.add_argument('--atol', type=float, default=0.01)
    parser.add_argument('--rtol', type=float, default=0.01)
    parser.add_argument('--min_mask_agreement', type=float, default=0.995)
    parser.add_argument('--max_increase_percent', type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.warmup_seconds < 3 or args.rounds < 5 or args.samples < 1000:
        parser.error('Use warmup_seconds >= 3, rounds >= 5 and samples >= 1000 for a latency conclusion.')
    if args.cpu_threads < 1 or args.workspace_mib < 1 or len(set(args.iters_s8)) != len(args.iters_s8):
        parser.error('Require positive threads/workspace and unique S8 iteration counts.')
    if not all(np.isfinite(v) for v in (args.warmup_seconds, args.atol, args.rtol,
                                        args.min_mask_agreement, args.max_increase_percent)):
        parser.error('Numeric benchmark settings must be finite.')
    if args.atol < 0 or args.rtol < 0 or not 0 <= args.min_mask_agreement <= 1 or args.max_increase_percent < 0:
        parser.error('Invalid logit tolerances, mask agreement or latency limit.')
    for iterations in args.iters_s8:
        validate_shape_iterations(args.img_size, args.iters_s16, iterations)
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    estimate = cost_estimate(args.img_size)
    (folder / 'estimate.json').write_text(json.dumps(estimate, indent=2) + '\n')
    if args.phase == 'estimate':
        print(json.dumps(estimate, indent=2))
        return
    # An interrupted rerun must never leave an old successful report looking current.
    (folder / 'report.json').write_text(json.dumps({'status': 'running', 'within_five_percent': None}, indent=2) + '\n')
    try:
        if args.phase == 'run' and (folder / 'manifest.json').exists():
            saved = json.loads((folder / 'manifest.json').read_text())
            args.img_size, args.iters_s16, args.iters_s8 = saved['img_size'], saved['iters_s16'], saved['iters_s8']
            for iterations in args.iters_s8:
                validate_shape_iterations(args.img_size, args.iters_s16, iterations)
            estimate = cost_estimate(args.img_size)
            (folder / 'estimate.json').write_text(json.dumps(estimate, indent=2) + '\n')
        if args.phase in ('prepare', 'all'):
            prepare(args)
        if args.phase == 'prepare':
            (folder / 'report.json').write_text(json.dumps({'status': 'prepared_not_measured',
                                                           'within_five_percent': None}, indent=2) + '\n')
            print(f'Artifacts ready in {folder}; run --phase run on the measuring GPU.')
            return
        report = run_benchmark(args)
    except (RuntimeError, ImportError, ValueError, OSError, KeyError) as error:
        report = dict(status='not_measured', reason=str(error), estimate=estimate,
                      within_five_percent=None)
        (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'No latency conclusion: {error}', file=sys.stderr)
        raise SystemExit(2) from error
    (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Latency status: {report["status"]}; report: {folder / "report.json"}')


if __name__ == '__main__':
    main()
