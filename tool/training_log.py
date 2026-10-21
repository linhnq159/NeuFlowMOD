"""Persistent epoch records and validation previews, independent of TensorBoard."""

import csv
import json
import logging
import math
import os
import platform
import shlex
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch


BEST_NAMES = ('best_iou', 'best_nIoU', 'best_PD', 'best_FA')
VAL_FIELDS = ('val_loss', 'val_iou', 'val_precision', 'val_recall',
              'val_target_fg_ratio', 'val_pred_fg_ratio', 'val_niou',
              'val_niou_positive', 'val_positive_samples', 'val_pd', 'val_fa')
HISTORY_FIELDS = (
    'session_id', 'epoch', 'loss_version', 'train_loss_total', 'train_loss_main',
    'train_loss_aux', 'train_loss_kd', 'train_iou', 'train_precision', 'train_recall',
    'train_target_fg_ratio', 'train_pred_fg_ratio', 'train_samples',
    *VAL_FIELDS, 'val_samples', 'lr_head', 'lr_backbone',
    'lr_head_first', 'lr_head_last', 'lr_backbone_first', 'lr_backbone_last',
    'backbone_training', 'train_seconds', 'val_seconds', 'epoch_seconds',
    'amp_scale', 'optimizer_steps', 'optimizer_skipped_steps', 'gpu_peak_memory_mb',
)


DETECTION_BEST_NAMES = ('best_map', 'best_map50')
DETECTION_VAL_FIELDS = ('val_loss', 'val_loss_center', 'val_loss_size', 'val_loss_offset',
                        'val_map', 'val_map50', 'val_precision', 'val_recall', 'val_boxes',
                        'val_collisions', 'val_empty_samples', 'val_empty_false_positives', 'val_empty_fp_per_image')
DETECTION_HISTORY_FIELDS = (
    'session_id', 'epoch', 'loss_version', 'train_loss_total', 'train_loss_center',
    'train_loss_size', 'train_loss_offset', 'train_boxes', 'train_collisions', 'train_samples',
    *DETECTION_VAL_FIELDS, 'val_samples', *HISTORY_FIELDS[HISTORY_FIELDS.index('lr_head'):],
)


def json_values(value):
    """Use plain metadata types, also loadable with weights_only on PyTorch 2.3."""
    if isinstance(value, dict):
        return {str(key): json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return str(value)  # torch.__version__ may be a TorchVersion subclass.
    if value is None:
        return None
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(json_values(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_alias(source, destination):
    """Atomically share checkpoint storage when supported; writers replace files."""
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


class SegmentationStats:
    """Pixel confusion counts and mean per-image IoU at sigmoid(logits) > 0.5."""

    def __init__(self):
        self.tp = self.pred = self.target = self.pixels = self.samples = 0
        self.niou_sum = self.positive_niou_sum = 0.0
        self.positive_samples = 0

    @torch.no_grad()
    def update(self, logits, labels):
        prediction, target = logits.detach() > 0, labels.detach() > 0
        dims = tuple(range(1, labels.ndim))
        counts = torch.stack(((prediction & target).sum(dim=dims),
                              prediction.sum(dim=dims), target.sum(dim=dims))).cpu()
        inter, pred, lab = counts.double()
        union = pred + lab - inter
        iou = inter / union.clamp_min(1)
        positive = lab > 0
        self.tp += int(inter.sum())
        self.pred += int(pred.sum())
        self.target += int(lab.sum())
        self.pixels += labels.numel()
        self.samples += labels.shape[0]
        self.niou_sum += float(iou.sum())
        self.positive_niou_sum += float(iou[positive].sum())
        self.positive_samples += int(positive.sum())

    def values(self):
        return dict(iou=self.tp / max(self.pred + self.target - self.tp, 1),
                    precision=self.tp / max(self.pred, 1), recall=self.tp / max(self.target, 1),
                    target_fg_ratio=self.target / max(self.pixels, 1),
                    pred_fg_ratio=self.pred / max(self.pixels, 1),
                    niou=self.niou_sum / max(self.samples, 1),
                    niou_positive=(self.positive_niou_sum / self.positive_samples
                                   if self.positive_samples else None),
                    positive_samples=self.positive_samples)


class RunHistory:
    def __init__(self, folder, args):
        self.folder = Path(folder)
        detection = getattr(args, "task", "segmentation") == "detection"
        self.fields = DETECTION_HISTORY_FIELDS if detection else HISTORY_FIELDS
        self.session_id = datetime.now().strftime('%Y%m%dT%H%M%S%f') + '-' + uuid.uuid4().hex[:8]
        self.session_dir = self.folder / 'sessions' / self.session_id
        self.session_dir.mkdir(parents=True)
        self.logger = logging.getLogger(f'training.{self.session_id}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in (logging.FileHandler(self.folder / 'train.log'), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.logger.addHandler(handler)
        self.snapshot = dict(
            session_id=self.session_id, started_at=datetime.now(timezone.utc).isoformat(),
            command=shlex.join(sys.argv), argv=list(sys.argv), cwd=str(Path.cwd()),
            python=platform.python_version(), python_executable=sys.executable,
            pytorch=str(torch.__version__), cuda=str(torch.version.cuda) if torch.version.cuda else None,
            cudnn=torch.backends.cudnn.version(), requested_config=json_values(vars(args)),
        )
        write_json(self.session_dir / 'session.json', self.snapshot)
        self.summary = dict(status='initializing', session_id=self.session_id,
                            last_completed_epoch=None, best={}, primary_checkpoint='checkpoint/best_map.pth' if detection else 'checkpoint/best_iou.pth')
        self._write_summary()
        self.logger.info('Session %s; command: %s', self.session_id, self.snapshot['command'])
        self.logger.info('Runtime: Python %s; PyTorch %s; CUDA %s',
                         self.snapshot['python'], self.snapshot['pytorch'], self.snapshot['cuda'])

    def configure(self, args, checkpoint, selection):
        self.snapshot.update(config=json_values(vars(args)),
                             resumed_checkpoint_epoch=checkpoint.get('epoch') if checkpoint else None,
                             resumed_session_id=checkpoint.get('session_id') if checkpoint else None,
                             source_loss_version=(checkpoint.get('loss_version', 'legacy_unversioned')
                                                  if checkpoint else None),
                             tb_image_samples=selection)
        write_json(self.session_dir / 'session.json', self.snapshot)
        write_json(self.session_dir / 'config.json', vars(args))
        # Original config stays intact; effective config is also saved per session.
        if not (self.folder / 'config.json').exists():
            write_json(self.folder / 'config.json', vars(args))
        self.summary['last_completed_epoch'] = checkpoint.get('epoch') if checkpoint else None
        self.summary['status'] = 'running'
        self._write_summary()
        self.logger.info('Effective config: %s', json.dumps(json_values(vars(args)), sort_keys=True))

    def _write_summary(self):
        write_json(self.session_dir / 'summary.json', self.summary)
        write_json(self.folder / 'summary.json', self.summary)

    def set_best(self, best):
        self.summary['best'] = best
        self._write_summary()

    def append(self, metrics, best):
        row = {name: json_values(metrics.get(name)) for name in self.fields}
        with (self.folder / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        path = self.folder / 'history.csv'
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        self.summary.update(last_completed_epoch=row['epoch'], best=best, latest_metrics=row)
        self._write_summary()
        self.logger.info('Epoch summary: %s', json.dumps(row, allow_nan=False))

    def finish(self, status, error=None):
        self.summary.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
        if error is not None:
            self.summary['error'] = str(error)
            self.logger.error('Training %s', status, exc_info=True)
        self._write_summary()
        self.logger.info('Session %s', status)

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def sample_identity(dataset, index):
    current, past, annotation = dataset.samples[index]
    identity = dict(current=str(current.relative_to(dataset.image_dir)), past=str(past.relative_to(dataset.image_dir)))
    if hasattr(dataset, 'label_dir'):
        identity['label'] = str(annotation.relative_to(dataset.label_dir))
    else:
        identity['mask'] = str(annotation.relative_to(dataset.mask_dir))
    return identity


def select_preview_samples(dataset, count, saved=()):
    """Prefer 3/4 positive masks and diverse sequences, retaining saved pairs.

    Only read/resize labels; no stabilization or model calls. Limit candidates
    per sequence, and stop scanning it once both category quotas are satisfied.
    """
    if dataset is None:
        return [], []
    identities = [sample_identity(dataset, index) for index in range(len(dataset))]
    key = lambda item: (item['current'], item['past'], item.get('mask', item.get('label')))
    lookup = {key(identity): index for index, identity in enumerate(identities)}
    selected = list(dict.fromkeys(lookup[key(item)] for item in saved if key(item) in lookup))[:count]
    if len(selected) == min(count, len(dataset)):
        return selected, [identities[index] for index in selected]
    groups = defaultdict(list)
    for index, (current, _, _) in enumerate(dataset.samples):
        groups[str(current.parent)].append(index)
    candidates = {True: [], False: []}
    positive_goal = max(1, round(count * 0.75))
    negative_goal = count - positive_goal
    for indices in groups.values():
        found = Counter()
        for index in indices:
            if hasattr(dataset, 'annotations'):
                positive = bool(len(dataset.annotations[index]))
            else:
                with Image.open(dataset.samples[index][2]) as image:
                    values = np.array(image)
                    mask = np.any(values[..., :3] > 0, axis=-1) if values.ndim == 3 else values > 0
                    if dataset.img_size is None:
                        with Image.open(dataset.samples[index][0]) as current:
                            size = current.size
                    else:
                        size = dataset.img_size
                    mask = Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
                    positive = bool(np.asarray(mask).any())
            if found[positive] < count:
                candidates[positive].append(index)
                found[positive] += 1
            if found[True] >= positive_goal and found[False] >= negative_goal:
                break
    used_sequences = Counter(str(dataset.samples[index][0].parent) for index in selected)

    def take(pool, amount):
        for _ in range(amount):
            remaining = [index for index in pool if index not in selected]
            if not remaining or len(selected) >= count:
                break
            index = min(remaining, key=lambda item: used_sequences[str(dataset.samples[item][0].parent)])
            selected.append(index)
            used_sequences[str(dataset.samples[index][0].parent)] += 1

    already_positive = sum(index in candidates[True] for index in selected)
    already_negative = sum(index in candidates[False] for index in selected)
    take(candidates[True], max(0, positive_goal - already_positive))
    take(candidates[False], max(0, negative_goal - already_negative))
    take(candidates[True] + candidates[False], count - len(selected))
    return selected, [identities[index] for index in selected]


def preview_panel(current, labels, logits):
    """Horizontal current RGB / target / prediction / probability strip."""
    probability = logits.detach().float().cpu().sigmoid()
    return torch.cat((current.detach().float().cpu().div(255).clamp(0, 1),
                      labels.detach().float().cpu().expand(3, -1, -1),
                      (logits.detach().cpu() > 0).float().expand(3, -1, -1),
                      probability.expand(3, -1, -1)), dim=2)
