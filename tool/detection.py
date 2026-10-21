"""Detection targets, loss, decoding and single-class all-area AP (101 recall points)."""

import math
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

DETECTION_LOSS_VERSION = 'centernet_v1'
DECODE_DEFAULTS = dict(conf_threshold=0.25, nms_iou=0.5, max_detections=100)
DETECTION_CONFIG = dict(head='centernet', stride=4, num_classes=1, hidden_channels=32,
                        label_format='yolo_txt', size_weight=0.1, offset_weight=1.,
                        heatmap_overlap=0.7, ap_score_floor=0.001, ap_recall_points=101)


def add_detection_arguments(parser):
    parser.add_argument('--task', choices=['segmentation', 'detection'], default=None,
                        help='New training: segmentation; resume/inference: checkpoint task')
    parser.add_argument('--conf_threshold', type=float, default=None, help='Detection score threshold (default 0.25)')
    parser.add_argument('--nms_iou', type=float, default=None, help='Detection NMS IoU (default 0.5)')
    parser.add_argument('--max_detections', type=int, default=None, help='Detection top-K / maximum outputs (default 100)')


def resolve_detection_options(args, saved=None):
    saved = saved or {}
    for name, default in DECODE_DEFAULTS.items():
        value = getattr(args, name, None)
        setattr(args, name, saved.get(name, default) if value is None else value)
    for name in ('conf_threshold', 'nms_iou'):
        value = getattr(args, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and between 0 and 1.')
    if type(args.max_detections) is not int or args.max_detections < 1:
        raise ValueError('max_detections must be a positive integer.')


def _radius(height, width, overlap=0.7):
    # CenterNet's three overlap constraints.
    b1, c1 = height + width, width * height * (1 - overlap) / (1 + overlap)
    b2, c2 = 2 * (height + width), (1 - overlap) * width * height
    a3, b3, c3 = 4 * overlap, -2 * overlap * (height + width), (overlap - 1) * width * height
    return max(0, int(min((b1 + math.sqrt(max(0., b1*b1 - 4*c1))) / 2,
                         (b2 + math.sqrt(max(0., b2*b2 - 16*c2))) / 2,
                         (b3 + math.sqrt(max(0., b3*b3 - 4*a3*c3))) / 2)))


def make_targets(boxes, height, width, device):
    heatmap = torch.zeros((height, width), device=device)
    targets = {}
    # CPU target construction avoids synchronizing a CUDA stream for every box.
    for box in boxes.detach().cpu().tolist():
        x1, y1, x2, y2 = [value / 4 for value in box]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x, y = int(cx), int(cy)
        if w <= 0 or h <= 0 or not (0 <= x < width and 0 <= y < height):
            raise ValueError('Detection target is degenerate or outside the feature map.')
        radius = _radius(h, w)
        left, right = min(x, radius), min(width - x - 1, radius)
        top, bottom = min(y, radius), min(height - y - 1, radius)
        yy = torch.arange(-top, bottom + 1, device=device, dtype=torch.float32)[:, None]
        xx = torch.arange(-left, right + 1, device=device, dtype=torch.float32)[None, :]
        gaussian = torch.exp(-(xx.square() + yy.square()) / (2 * ((2 * radius + 1) / 6)**2))
        patch = heatmap[y-top:y+bottom+1, x-left:x+right+1]
        torch.maximum(patch, gaussian, out=patch)
        key = (y, x)
        if key not in targets or w * h < targets[key][0] * targets[key][1]:
            targets[key] = (w, h, cx - x, cy - y)
    return heatmap, targets, len(boxes) - len(targets)


class DetectionLoss(nn.Module):
    def forward(self, outputs, labels):
        logits, sizes, offsets = (outputs[k].float() for k in ('center_logits', 'size', 'offset'))
        center_losses, size_losses, offset_losses = [], [], []
        collisions = 0
        for i, boxes in enumerate(labels):
            target, objects, count = make_targets(boxes, *logits.shape[-2:], logits.device)
            collisions += count
            logit = logits[i, 0]
            p = logit.sigmoid()
            positive = target == 1
            focal = -F.logsigmoid(logit) * (1-p).square() * positive
            focal -= F.logsigmoid(-logit) * p.square() * (1-target).pow(4) * ~positive
            center_losses.append(focal.sum() / max(len(objects), 1))
            if objects:
                indices = torch.tensor(list(objects), device=logits.device, dtype=torch.long)
                values = logits.new_tensor(list(objects.values()))
                yy, xx = indices.unbind(1)
                size_losses.append(F.l1_loss(sizes[i, :, yy, xx].T, values[:, :2], reduction='sum') / len(objects))
                offset_losses.append(F.l1_loss(offsets[i, :, yy, xx].T, values[:, 2:], reduction='sum') / len(objects))
            else:
                size_losses.append(sizes[i].sum() * 0)
                offset_losses.append(offsets[i].sum() * 0)
        center, size, offset = [torch.stack(values).mean() for values in (center_losses, size_losses, offset_losses)]
        return dict(total=center + .1 * size + offset, center=center, size=size, offset=offset,
                    collisions=collisions, boxes=sum(len(boxes) for boxes in labels))


@torch.no_grad()
def decode_detections(outputs, input_size, original_size=None, conf_threshold=.25, nms_iou=.5, max_detections=100):
    """Return one [N,6] xyxy, score, class tensor per image; sizes are (width,height)."""
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError('Model produced non-finite detection outputs.')
    width, height = input_size
    logits, size, offset = [outputs[k].float() for k in ('center_logits', 'size', 'offset')]
    heatmap = logits.sigmoid()
    heatmap = heatmap * (heatmap == F.max_pool2d(heatmap, 3, stride=1, padding=1))
    h, w = heatmap.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device), torch.arange(w, device=logits.device), indexing='ij')
    result = []
    for i in range(len(logits)):
        cx = (xx + offset[i, 0]) * 4
        cy = (yy + offset[i, 1]) * 4
        valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        scores = heatmap[i, 0].masked_fill(~valid, -1).flatten()
        scores, indices = torch.topk(scores, min(max_detections, scores.numel()))
        selected = scores > conf_threshold
        scores, indices = scores[selected], indices[selected]
        centers = torch.stack((cx.flatten()[indices], cy.flatten()[indices]), 1)
        wh = size[i].reshape(2, -1)[:, indices].T * 4
        boxes = torch.cat((centers - wh / 2, centers + wh / 2), 1)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
        boxes, scores = boxes[keep], scores[keep]
        keep = nms(boxes, scores, nms_iou)[:max_detections]
        boxes, scores = boxes[keep], scores[keep]
        if original_size is not None:
            ow, oh = original_size
            boxes = boxes * boxes.new_tensor([ow / width, oh / height, ow / width, oh / height])
        result.append(torch.cat((boxes, scores[:, None], torch.zeros_like(scores[:, None])), 1))
    return result


def detection_records(detections):
    return [dict(bbox_xyxy=row[:4], score=row[4], class_id=int(row[5]))
            for row in detections.detach().cpu().tolist()]


def draw_detections(rgb, detections, color=(255, 0, 0), thickness=2, scores=True):
    image = np.asarray(rgb).copy()
    for row in detections.detach().cpu().tolist():
        x1, y1, x2, y2 = row[:4]
        start = (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))))
        end = (min(image.shape[1]-1, int(math.ceil(x2))-1), min(image.shape[0]-1, int(math.ceil(y2))-1))
        cv2.rectangle(image, start, end, color, thickness)
        if scores and len(row) > 4:
            cv2.putText(image, f'{row[4]:.2f}', (start[0], max(12, start[1]-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return image


class DetectionStats:
    def __init__(self, conf_threshold=.25):
        self.records = []
        self.matches = []
        self.conf_threshold = conf_threshold

    def update(self, predictions, labels):
        for prediction, target in zip(predictions, labels):
            prediction, target = prediction.detach().float().cpu(), target.detach().float().cpu()
            prediction = prediction[torch.argsort(prediction[:, 4], descending=True, stable=True)]
            self.records.append((prediction, target))
            self.matches.append((prediction[:, 4].numpy(), box_iou(prediction[:, :4].double(), target.double()).numpy()))

    def _match(self, threshold):
        scores, correct = [], []
        for image_scores, ious in self.matches:
            if ious.shape[1] == 0:
                scores.extend(image_scores)
                correct.extend([False] * len(image_scores))
                continue
            used = np.zeros(ious.shape[1], dtype=bool)
            for score, overlaps in zip(image_scores, ious):
                overlaps = overlaps.copy()
                overlaps[used] = -1
                hit = bool(overlaps.max() >= threshold)
                if hit:
                    used[int(overlaps.argmax())] = True
                scores.append(float(score))
                correct.append(hit)
        order = np.argsort(-np.asarray(scores), kind='stable')
        return np.asarray(scores)[order], np.asarray(correct, dtype=np.float64)[order]

    def values(self):
        targets = sum(len(target) for _, target in self.records)
        aps = []
        recall_grid = np.linspace(0, 1, 101)
        for threshold in np.linspace(.5, .95, 10):
            scores, hits = self._match(threshold)
            tp = np.cumsum(hits[scores > .001])
            recall = tp / max(targets, 1)
            precision = tp / np.maximum(np.arange(len(tp)) + 1, 1)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            indices = np.searchsorted(recall, recall_grid, side='left')
            aps.append(float(np.append(envelope, 0.)[indices].mean()))
            if threshold == .5:
                selected = scores > self.conf_threshold
                true = float(hits[selected].sum())
                p = true / max(int(selected.sum()), 1)
                r = true / targets if targets else None
        empty = [(pred, target) for pred, target in self.records if not len(target)]
        false_boxes = sum(int((pred[:, 4] > self.conf_threshold).sum()) for pred, _ in empty)
        return dict(map=float(np.mean(aps)) if targets else None, map50=aps[0] if targets else None,
                    precision=p, recall=r, boxes=targets, empty_samples=len(empty),
                    empty_false_positives=false_boxes,
                    empty_fp_per_image=false_boxes / len(empty) if empty else None)
