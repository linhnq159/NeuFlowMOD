"""Checkpoint loading shared by the segmentation trainer and inference CLI."""

from argparse import Namespace

import torch


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


def load_model_weights(model, checkpoint, strict=True):
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    if not set(state).intersection(model.state_dict()):
        raise ValueError('No matching model weights in checkpoint.')
    return model.load_state_dict(state, strict=strict)


def load_weights(model, path, strict=True):
    result = load_model_weights(model, load_checkpoint(path), strict=strict)
    print(f'Loaded {path} ({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys)')
