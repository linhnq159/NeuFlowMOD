"""Task-specific epoch logic; checkpoint/session/optimizer lifecycle is shared with segmentation."""

import time
import numpy as np
import torch
from tqdm import tqdm

from tool.detection import DetectionStats, decode_detections, draw_detections


def train_detection(trainer, epoch):
    t, args = trainer, trainer.args
    started = time.perf_counter()
    if t.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(t.device)
    t.model.train()
    t._set_backbone_training(epoch)
    if args.optim == 'msam':
        t.optimizer.move_up_to_momentumAscent()
    totals = dict(total=0., center=0., size=0., offset=0.)
    samples = boxes = collisions = updates = skipped = 0
    lr_values = []
    progress = tqdm(t.train_data_loader)
    for current, past, labels in progress:
        images = [image.to(t.device, non_blocking=True) for image in (current, past)]
        t.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=t.device.type, enabled=t.scaler.is_enabled()):
            outputs = t._forward(t.model, images)
        losses = t.criterion(outputs, labels)
        loss = losses['total']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite detection loss at epoch {epoch}.')
        t.scaler.scale(loss).backward()
        if args.clip_grad:
            t.scaler.unscale_(t.optimizer)
            torch.nn.utils.clip_grad_norm_(t.model.parameters(), args.clip_grad)
        old_scale = t.scaler.get_scale()
        rates = [group['lr'] for group in t.optimizer.param_groups]
        t.scaler.step(t.optimizer)
        t.scaler.update()
        batch_size = len(labels)
        if t.scaler.get_scale() >= old_scale:
            updates += 1
            lr_values.append((rates[0], rates[1] if t.backbone_training else 0., batch_size))
            if args.lr_sch == 'ilnet':
                t.lr_scheduler.step()
        else:
            skipped += 1
        for name in totals:
            totals[name] += losses[name].item() * batch_size
        samples += batch_size
        boxes += losses['boxes']
        collisions += losses['collisions']
        progress.set_description(f'Epoch {epoch}: detection loss={totals["total"]/samples:.5f}, lr_head={rates[0]:.3g}')
    if args.optim == 'msam':
        t.optimizer.move_back_from_momentumAscent()
    if args.lr_sch == 'cos' and updates:
        t.lr_scheduler.step()
    if not samples:
        raise RuntimeError('Training loader produced no samples.')
    t.train_epoch_stats = dict(epoch=epoch, **{f'train_loss_{key}': value / samples for key, value in totals.items()},
                               train_samples=samples, train_boxes=boxes, train_collisions=collisions,
                               backbone_training=t.backbone_training, train_seconds=time.perf_counter()-started,
                               amp_scale=t.scaler.get_scale(), optimizer_steps=updates, optimizer_skipped_steps=skipped)
    used_samples = sum(row[2] for row in lr_values)
    for group, index in (('head', 0), ('backbone', 1)):
        t.train_epoch_stats[f'lr_{group}'] = sum(row[index]*row[2] for row in lr_values) / max(used_samples, 1)
        t.train_epoch_stats[f'lr_{group}_first'] = lr_values[0][index] if lr_values else 0.
        t.train_epoch_stats[f'lr_{group}_last'] = lr_values[-1][index] if lr_values else 0.
    return totals['total'] / samples


def detection_preview(current, boxes, predictions):
    rgb = current.detach().cpu().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
    gt = draw_detections(rgb, boxes, color=(0, 255, 0), scores=False)
    pred = draw_detections(rgb, predictions)
    return torch.from_numpy(np.concatenate((rgb, gt, pred), axis=1)).permute(2, 0, 1).float() / 255


@torch.no_grad()
def validate_detection(trainer, epoch):
    t, args = trainer, trainer.args
    started = time.perf_counter()
    t.model.eval()
    stats = DetectionStats(args.conf_threshold)
    totals = dict(total=0., center=0., size=0., offset=0.)
    samples = collisions = 0
    panels = {}
    selected = set(t.tb_sample_indices) if t.writer is not None else set()
    for current, past, labels in tqdm(t.val_data_loader, desc=f'Validation {epoch}'):
        images = [image.to(t.device, non_blocking=True) for image in (current, past)]
        with torch.autocast(device_type=t.device.type, enabled=t.scaler.is_enabled()):
            outputs = t._forward(t.model, images)
        losses = t.criterion(outputs, labels)
        if not torch.isfinite(losses['total']):
            raise RuntimeError(f'Non-finite validation loss at epoch {epoch}.')
        predictions = decode_detections(outputs, (current.shape[-1], current.shape[-2]),
                                        conf_threshold=min(.001, args.conf_threshold), nms_iou=args.nms_iou,
                                        max_detections=args.max_detections)
        stats.update(predictions, labels)
        for offset, (boxes, prediction) in enumerate(zip(labels, predictions)):
            index = samples + offset
            if index in selected:
                panels[index] = detection_preview(current[offset], boxes, prediction[prediction[:, 4] > args.conf_threshold])
        for key in totals:
            totals[key] += losses[key].item() * len(labels)
        samples += len(labels)
        collisions += losses['collisions']
    if not samples:
        raise RuntimeError('Validation loader produced no samples.')
    values = stats.values()
    t.val_epoch_stats = dict(epoch=epoch, val_loss=totals['total']/samples,
                             **{f'val_loss_{key}': totals[key]/samples for key in ('center', 'size', 'offset')},
                             **{f'val_{key}': value for key, value in values.items()},
                             val_samples=samples, val_collisions=collisions, val_seconds=time.perf_counter()-started)
    improved = []
    for key in ('map', 'map50'):
        name, value = 'best_' + key, values[key]
        if value is not None and value > getattr(t, name):
            setattr(t, name, value)
            t.best_epochs[name] = epoch
            improved.append(name)
    t.validation_count += 1
    if (t.writer is not None and t.last_tb_image_epoch != epoch
            and (t.validation_count == 1 or epoch % args.tb_image_freq == 0 or 'best_map' in improved)):
        for position, index in enumerate(t.tb_sample_indices):
            if index in panels:
                t.writer.add_image(f'Validation/sample_{position:02d}_RGB_GT_boxes_predictions', panels[index], epoch)
        t.last_tb_image_epoch = epoch
    return improved
