"""Checkpoint loading shared by the segmentation trainer and inference CLI."""

from argparse import Namespace

import torch
from NeuFlow import config


def load_checkpoint(path):
    # Earlier checkpoints may contain argparse.Namespace.
    with torch.serialization.safe_globals([Namespace]):
        return torch.load(path, map_location='cpu', weights_only=True)


def checkpoint_args(checkpoint):
    args = checkpoint.get('args', {}) if checkpoint is not None else {}
    return vars(args) if isinstance(args, Namespace) else args


def resolve_stabilization(override, checkpoint=None):
    mode = override
    if mode is None:
        mode = ('homography' if checkpoint is None
                else checkpoint_args(checkpoint).get('stabilization', 'none'))
    if mode not in ('none', 'homography'):
        raise ValueError(f'Unsupported stabilization: {mode}')
    return mode


def resolve_seg_input(override=None, checkpoint=None):
    """A checkpoint owns its architecture; legacy checkpoints always mean flow."""
    if checkpoint is None:
        return config.validate_seg_input(config.seg_input if override is None else override)
    saved = config.validate_seg_input(checkpoint_args(checkpoint).get('seg_input', 'flow'))
    if override is not None and config.validate_seg_input(override) != saved:
        raise ValueError(f'seg_input={override!r} conflicts with checkpoint seg_input={saved!r}. '
                         'Use --transfer when changing the training architecture; not --resume/inference.')
    return saved


def checkpoint_state(checkpoint):
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    return {key.removeprefix('module.'): value for key, value in state.items()}


def load_model_weights(model, checkpoint, strict=True):
    if hasattr(model, 'seg_input'):
        resolve_seg_input(model.seg_input, checkpoint)
    state = checkpoint_state(checkpoint)
    if not set(state).intersection(model.state_dict()):
        raise ValueError('No matching model weights in checkpoint.')
    return model.load_state_dict(state, strict=strict)


def transfer_model_weights(model, checkpoint):
    """Load weights only; allow flow -> flow_rgb with an initially silent RGB input."""
    source = resolve_seg_input(checkpoint=checkpoint)
    target = model.seg_input
    if source == target:
        return load_model_weights(model, checkpoint)
    if (source, target) != ('flow', 'flow_rgb'):
        raise ValueError(f'Unsupported seg_input transfer: {source} -> {target}.')
    state = checkpoint_state(checkpoint)
    key = 'seg_head.initial_block.main_branch.weight'
    old_weight = state.get(key)
    expected = model.state_dict()[key]
    if old_weight is None or tuple(old_weight.shape) != (6, 2, 3, 3) or tuple(expected.shape) != (6, 5, 3, 3):
        raise ValueError('flow -> flow_rgb transfer requires a full ModelBuilder checkpoint with a 6x2x3x3 stem.')
    expanded = old_weight.new_zeros(expected.shape)
    expanded[:, :2].copy_(old_weight)
    state[key] = expanded
    result = model.load_state_dict(state, strict=True)
    print('Transferred flow -> flow_rgb: copied flow weights; initialized RGB weights to zero.')
    return result


def load_weights(model, path, strict=True):
    result = load_model_weights(model, load_checkpoint(path), strict=strict)
    print(f'Loaded {path} ({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys)')
