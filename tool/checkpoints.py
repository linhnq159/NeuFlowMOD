"""Checkpoint loading shared by the segmentation trainer and inference CLI."""

from argparse import Namespace
from contextlib import nullcontext

import torch
from NeuFlow import config
from data_utils.frame_augmentation import validate_augmentation


def load_checkpoint(path):
    # PyTorch 2.4+ can allowlist Namespace metadata from older checkpoints.
    # On 2.3, plain state dicts and our dict-based training checkpoints still
    # load with weights_only=True; never retry with unrestricted unpickling.
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    context = safe_globals([Namespace]) if safe_globals is not None else nullcontext()
    with context:
        return torch.load(path, map_location='cpu', weights_only=True)


def checkpoint_args(checkpoint):
    args = checkpoint.get('args', {}) if checkpoint is not None else {}
    return vars(args) if isinstance(args, Namespace) else args


def resolve_task(override=None, checkpoint=None):
    saved = checkpoint_args(checkpoint).get('task', 'segmentation')
    task = saved if checkpoint is not None or override is None else override
    if task not in ('segmentation', 'detection'):
        raise ValueError(f'Unsupported task: {task}')
    if checkpoint is not None and override is not None and override != task:
        raise ValueError(f'task={override!r} conflicts with checkpoint task={task!r}; use --transfer for a new task.')
    return task


def resolve_train_augmentation(override=None, checkpoint=None):
    """Resume owns numeric settings; explicit CLI choices replace only enabled modes."""
    settings = (config.train_augmentation if checkpoint is None else
                checkpoint_args(checkpoint).get('train_augmentation', config.DEFAULT_TRAIN_AUGMENTATION))
    settings = dict(settings)
    if override is not None:
        if 'none' in override and list(override) != ['none']:
            raise ValueError('--augment none cannot be combined with other augmentations.')
        settings['enabled'] = [] if list(override) == ['none'] else list(override)
    return validate_augmentation(settings)


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
    if hasattr(model, 'task'):
        resolve_task(model.task, checkpoint)
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
    source_task, target_task = resolve_task(checkpoint=checkpoint), getattr(model, 'task', 'segmentation')
    if source_task != target_task and (source_task, target_task) != ('segmentation', 'detection'):
        raise ValueError(f'Unsupported task transfer: {source_task} -> {target_task}.')
    if source == target and source_task == target_task:
        return load_model_weights(model, checkpoint)
    if source != target and (source, target) != ('flow', 'flow_rgb'):
        raise ValueError(f'Unsupported seg_input transfer: {source} -> {target}.')
    state = checkpoint_state(checkpoint)
    if source != target:
        key = 'seg_head.initial_block.main_branch.weight'
        old_weight, expected = state.get(key), model.state_dict()[key]
        if old_weight is None or tuple(old_weight.shape) != (6, 2, 3, 3) or tuple(expected.shape) != (6, 5, 3, 3):
            raise ValueError('flow -> flow_rgb transfer requires a full ModelBuilder checkpoint with a 6x2x3x3 stem.')
        expanded = old_weight.new_zeros(expected.shape)
        expanded[:, :2].copy_(old_weight)
        state[key] = expanded
        print('Transferred flow -> flow_rgb: copied flow weights; initialized RGB weights to zero.')
    if source_task != target_task:
        # Fill only the new head; all shared weights must match strictly.
        state.update({key: value for key, value in model.state_dict().items() if key.startswith('detection_head.')})
        print('Transferred segmentation -> detection: shared weights loaded; detection head initialized.')
    result = model.load_state_dict(state, strict=True)
    return result


def load_weights(model, path, strict=True):
    result = load_model_weights(model, load_checkpoint(path), strict=strict)
    print(f'Loaded {path} ({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys)')
