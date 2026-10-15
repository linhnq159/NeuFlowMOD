"""Export a full frame-pair MOD model with fixed shapes and verify ONNX logits."""

import copy
import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from NeuFlow.model_builder import ModelBuilder
from external.ILNet.model.LW_Opticalflow import SEBlock
from tool.checkpoints import checkpoint_args, load_checkpoint, load_model_weights, resolve_seg_input


class HalfScale(nn.Module):
    """Exact eval equivalent of a bias-free SE block with zero hidden channels."""

    def forward(self, x):
        return x * 0.5


def fold_empty_se(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SEBlock) and child.fc[0].out_features == 0:
            setattr(module, name, HalfScale())
            count += 1
        else:
            count += fold_empty_se(child)
    return count


class ExportModel(nn.Module):
    def __init__(self, model, width, height, iters_s16=1, iters_s8=8):
        super().__init__()
        validate_shape_iterations((width, height), iters_s16, iters_s8)
        # Never mutate a training model/checkpoint for export.
        self.model = copy.deepcopy(model).cpu().float().eval()
        self.folded_se_blocks = fold_empty_se(self.model)
        self.iters_s16, self.iters_s8 = iters_s16, iters_s8
        self.model.backbone_op.init_bhwd(1, height, width, torch.device('cpu'), amp=False)
        self.model._flow_init_key = (1, height, width, torch.device('cpu'), False)

    def forward(self, current, past):
        return self.model(current, past, self.iters_s16, self.iters_s8)


def validate_shape_iterations(img_size, iters_s16, iters_s8):
    if len(img_size) != 2 or any(type(n) is not int or n < 32 or n % 16 for n in img_size):
        raise ValueError('img_size WIDTH HEIGHT must be integers >= 32 and divisible by 16.')
    if any(type(n) is not int or n < 1 for n in (iters_s16, iters_s8)):
        raise ValueError('S16/S8 iterations must be positive integers.')


def validation_pairs(width, height, seed=1211):
    generator = torch.Generator().manual_seed(seed)
    # Independent frames and a mildly displaced pair exercise different matching regimes.
    current = torch.rand((1, 3, height, width), generator=generator) * 255
    past = torch.rand(current.shape, generator=generator) * 255
    shifted = torch.roll(current, shifts=(1, 2), dims=(-2, -1))
    return [(current, past), (current, shifted)]


def compare_logits(reference, actual, atol=1e-3, rtol=1e-3):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Logit validation failed: shape mismatch or non-finite values.')
    error = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    passed = bool(np.allclose(reference, actual, atol=atol, rtol=rtol))
    return dict(passed=passed, max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                mask_agreement=float(np.mean((reference > 0) == (actual > 0))), atol=atol, rtol=rtol)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(model, output, img_size=(960, 544), iters_s16=1, iters_s8=8, seed=1211):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError('Export requires onnx and onnxruntime; see tool/requirements-export.txt.') from error
    validate_shape_iterations(img_size, iters_s16, iters_s8)
    output = Path(output)
    if output.suffix != '.onnx':
        raise ValueError('Export output must end in .onnx.')
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = img_size
    wrapper = ExportModel(model, width, height, iters_s16, iters_s8)
    pairs = validation_pairs(width, height, seed)
    references = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for current, past in pairs:
            references.append(wrapper(current, past).numpy())
        temporary = output.with_suffix('.partial.onnx')
        try:
            torch.onnx.export(wrapper, pairs[0], str(temporary), opset_version=16,
                              input_names=['current', 'past'], output_names=['logits'],
                              dynamo=False, do_constant_folding=True)
            graph = onnx.load(str(temporary))
            onnx.checker.check_model(graph)
            options = ort.SessionOptions()
            options.intra_op_num_threads = torch.get_num_threads()
            session = ort.InferenceSession(str(temporary), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            checks = []
            for pair, reference in zip(pairs, references):
                current, past = pair
                actual = session.run(['logits'], {'current': current.numpy(), 'past': past.numpy()})[0]
                check = compare_logits(reference, actual)
                checks.append(check)
                if not check['passed']:
                    raise RuntimeError(f'PyTorch/ONNX logit validation failed: {check}')
            del session
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    validation_path = output.with_suffix('.validation.npz')
    samples = {}
    for index, ((current, past), reference) in enumerate(zip(pairs, references)):
        samples.update({f'current_{index}': current.numpy(), f'past_{index}': past.numpy(),
                        f'logits_{index}': reference})
    np.savez(validation_path, **samples)
    metadata = dict(seg_input=model.seg_input, img_size=list(img_size), batch_size=1,
                    iters_s16=iters_s16, iters_s8=iters_s8, opset=16,
                    input_dtype='float32', input_range=[0, 255], output='float32 logits',
                    warp='external; not part of graph', folded_se_blocks=wrapper.folded_se_blocks,
                    pytorch_version=torch.__version__, onnx_version=onnx.__version__,
                    onnxruntime_version=ort.__version__, validation=checks,
                    validation_file=validation_path.name, validation_sha256=sha256_file(validation_path),
                    onnx_sha256=sha256_file(output))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported and verified {output} ({model.seg_input}, S8={iters_s8})')
    return metadata


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seg_input', choices=['flow', 'flow_rgb'])
    parser.add_argument('--img_size', nargs=2, type=int)
    parser.add_argument('--iters_s16', type=int)
    parser.add_argument('--iters_s8', type=int)
    parser.add_argument('--cpu_threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('cpu_threads must be positive.')
    if Path(args.checkpoint).resolve() == Path(args.output).resolve():
        parser.error('Output must not overwrite the checkpoint.')
    torch.set_num_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.checkpoint)
    saved = checkpoint_args(checkpoint)
    mode = resolve_seg_input(args.seg_input, checkpoint)
    model = ModelBuilder(seg_input=mode).eval()
    load_model_weights(model, checkpoint)
    export_model(model, args.output, args.img_size or saved.get('img_size', [960, 544]),
                 args.iters_s16 if args.iters_s16 is not None else saved.get('iters_s16', 1),
                 args.iters_s8 if args.iters_s8 is not None else saved.get('iters_s8', 8))


if __name__ == '__main__':
    main()
