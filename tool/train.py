"""Train frame-pair motion segmentation (masks) or detection (YOLO TXT boxes)."""

import json
import math
import random
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external' / 'ILNet'))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.frame_mask_dataset import FrameMaskDataset, collate_frame_masks
from data_utils.frame_box_dataset import FrameBoxDataset, collate_frame_boxes
from data_utils.frame_pair import init_frame_worker
from data_utils.frame_augmentation import AUGMENT_MODES
from tool.checkpoints import (load_checkpoint, load_weights, load_model_weights,
                              resolve_stabilization, resolve_seg_input, resolve_train_augmentation,
                              transfer_model_weights, resolve_task, checkpoint_args)
from tool.detection import (add_detection_arguments, resolve_detection_options, DetectionLoss,
                            DETECTION_LOSS_VERSION, DETECTION_CONFIG)
from NeuFlow.config import SEG_INPUT_MODES
from NeuFlow.model_builder import ModelBuilder
from external.ILNet.utils.Adan import Adan
from external.ILNet.optimizer.adamW_msam import AdamW_MSAM
from external.ILNet.utils.lr_scheduler import create_lr_scheduler
from external.ILNet.model.loss import SoftIoULoss, criterion, dice_critetion
from external.ILNet.model.metrics import PD_FA
from external.ILNet.loss.custom_loss import CustomLoss, CUSTOM_LOSS_VERSION
from tool.training_log import (BEST_NAMES, VAL_FIELDS, RunHistory, SegmentationStats,
                               checkpoint_alias, json_values, preview_panel, select_preview_samples)
from tool.training_log import DETECTION_BEST_NAMES, DETECTION_VAL_FIELDS


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    add_detection_arguments(parser)
    parser.add_argument('--work_dir', default='work_seg/checkpoint')
    parser.add_argument('--resize', '--img_size', dest='img_size', nargs=2, type=int,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Resize both frames and labels; omitted: native frames, masks match frames, boxes use frame coordinates')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--warm_up_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optim', choices=['adan', 'msam'], default='adan')
    parser.add_argument('--lr_sch', choices=['ilnet', 'cos'], default='ilnet')
    parser.add_argument('--seg_input', choices=SEG_INPUT_MODES, default=None,
                        help='Head input; new runs use NeuFlow/config.py, resume uses checkpoint')
    parser.add_argument('--data_path', help='Dataset root with images/ and masks/ (segmentation) or labels/ (detection)')
    parser.add_argument('--image_dir', '--train_image_dir', dest='image_dir', help='Training image folder')
    parser.add_argument('--mask_dir', '--train_mask_dir', dest='mask_dir', help='Training ground-truth mask folder')
    parser.add_argument('--val_image_dir', help='Optional separate validation image folder')
    parser.add_argument('--val_mask_dir', help='Optional separate validation mask folder')
    parser.add_argument('--label_dir', help='Training YOLO TXT folder; detection only')
    parser.add_argument('--val_label_dir', help='Separate validation YOLO TXT folder')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Last fraction of each sequence reserved for validation; 0 disables it')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Distance in sorted frames: inputs (t, t-frame_skip), target annotation at t')
    parser.add_argument('--iters_s16', type=int, default=1, help='Number of S16 refinement iterations')
    parser.add_argument('--iters_s8', type=int, default=8, help='Number of S8 refinement iterations')
    parser.add_argument('--stabilization', choices=['none', 'homography'], default=None,
                        help='Past-to-current alignment; new runs default to homography, resume uses checkpoint')
    parser.add_argument('--augment', nargs='+', choices=('none',) + AUGMENT_MODES, default=None,
                        help='Training augmentations; defaults from config or resume checkpoint. Use none to disable')
    parser.add_argument('--mode', default='L')
    parser.add_argument('--loss', type=str.upper, default=None,
                        choices=['BCE', 'SOFTIOULOSS', 'DICE', 'FOCAL', 'CUSTOM', 'ASIMMUNIFOCAL', 'NAF'])
    parser.add_argument('--resume', default='', help='Resume model, optimizer, scheduler and epoch')
    parser.add_argument('--transfer', default='', help='Initialize model weights from a checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--amp', action=BooleanOptionalAction, default=True)
    parser.add_argument('--tensorboard', action=BooleanOptionalAction, default=True)
    parser.add_argument('--tb_image_freq', type=int, default=10,
                        help='Preview images at first validation, every N epochs, and new best IoU')
    parser.add_argument('--tb_num_images', type=int, default=4,
                        help='Number of fixed validation previews (prefer 3 foreground / 1 empty)')
    parser.add_argument('--teacher', default='', help='Optional segmentation checkpoint for distillation; segmentation only')
    parser.add_argument('--alpha', type=float, default=0.5, help='Supervised loss weight when using a teacher')
    parser.add_argument('--T', type=float, default=10, help='Distillation temperature')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Keep a full checkpoint every N completed epochs, regardless of metrics; 0 disables')
    parser.add_argument('--gain', type=float, default=0.4,
                        help='Deprecated compatibility option; ignored by corrected CUSTOM loss')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='Gradient norm limit; 0 disables clipping')
    parser.add_argument('--seed', type=int, default=1211)
    parser.add_argument('--backbone_op_pretrained', default=str(ROOT / 'neuflow_sintel.pth'),
                        help='Pretrained NeuFlow weights; pass an empty string to initialize randomly')
    parser.add_argument('--epoch_training_backbone', type=int, default=300,
                        help='First epoch training the backbone: epochs 1..299 are frozen by default')
    parser.add_argument('--backbone_layer_lr', type=float, default=0.1,
                        help='Backbone LR / head LR throughout training')
    args = parser.parse_args(argv)
    if args.augment is not None and ('none' in args.augment and args.augment != ['none']
                                     or len(set(args.augment)) != len(args.augment)):
        parser.error('--augment requires unique choices; none cannot be combined with other modes.')
    if args.resume and args.transfer:
        parser.error('Choose either --resume or --transfer, not both.')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    try:
        args.task = resolve_task(args.task, checkpoint)
        resolve_detection_options(args, checkpoint_args(checkpoint))
    except ValueError as error:
        parser.error(str(error))
    if args.task == 'detection':
        if args.loss is not None or args.teacher:
            parser.error('Detection uses centernet_v1; --loss and segmentation --teacher are not supported.')
        args.loss = 'CENTERNET'
        args.detection_config = dict(DETECTION_CONFIG)
    else:
        args.loss = args.loss or 'CUSTOM'
    if args.batch_size < 2:
        parser.error('--batch_size must be >= 2 because the head has BatchNorm after global pooling.')
    if args.epochs < 1 or args.start_epoch < 1 or args.start_epoch > args.epochs:
        parser.error('Require 1 <= start_epoch <= epochs.')
    if args.frame_skip < 1 or args.eval_freq < 1 or args.num_workers < 0:
        parser.error('frame_skip/eval_freq must be >= 1 and num_workers must be >= 0.')
    if args.save_freq < 0:
        parser.error('--save_freq must be >= 0 (0 disables periodic checkpoints).')
    if args.tb_image_freq < 1 or args.tb_num_images < 1:
        parser.error('tb_image_freq and tb_num_images must be positive integers.')
    if args.iters_s16 < 1 or args.iters_s8 < 1:
        parser.error('iters_s16 and iters_s8 must be positive integers.')
    if args.learning_rate <= 0 or args.backbone_layer_lr <= 0 or args.clip_grad < 0:
        parser.error('Learning rates must be positive and clip_grad must be >= 0.')
    if args.epoch_training_backbone < 1:
        parser.error('--epoch_training_backbone must be >= 1.')
    if args.lr_sch == 'ilnet' and not 0 <= args.warm_up_epochs < args.epochs:
        parser.error('Require 0 <= warm_up_epochs < epochs; use --warm_up_epochs 0 for short runs.')
    if not 0 <= args.val_split < 1 or not 0 <= args.alpha <= 1 or args.T <= 0:
        parser.error('Require 0 <= val_split < 1, 0 <= alpha <= 1, and T > 0.')
    if args.img_size is not None and min(args.img_size) < 32:
        parser.error('--resize WIDTH HEIGHT must be >= 32.')
    label_arg = 'label_dir' if args.task == 'detection' else 'mask_dir'
    if bool(args.val_image_dir) != bool(getattr(args, 'val_' + label_arg)):
        parser.error(f'Provide both --val_image_dir and --val_{label_arg}.')
    if args.data_path:
        args.image_dir = args.image_dir or str(Path(args.data_path) / 'images')
        setattr(args, label_arg, getattr(args, label_arg) or str(Path(args.data_path) / ('labels' if args.task == 'detection' else 'masks')))
    if not args.image_dir or not getattr(args, label_arg):
        parser.error(f'Provide --image_dir and --{label_arg}, or --data_path.')
    return args


def build_datasets(args):
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    args.seg_input = resolve_seg_input(args.seg_input, checkpoint)
    args.task = resolve_task(getattr(args, 'task', None), checkpoint)
    if args.stabilization is None:
        args.stabilization = resolve_stabilization(None, checkpoint)
    args.train_augmentation = resolve_train_augmentation(getattr(args, 'augment', None), checkpoint)
    options = dict(img_size=args.img_size, frame_skip=args.frame_skip, stabilization=args.stabilization)
    dataset_class = FrameBoxDataset if args.task == 'detection' else FrameMaskDataset
    label_dir = args.label_dir if args.task == 'detection' else args.mask_dir
    val_label_dir = args.val_label_dir if args.task == 'detection' else args.val_mask_dir
    if args.val_image_dir:
        train_set = dataset_class(args.image_dir, label_dir, augmentation=args.train_augmentation, **options)
        val_set = dataset_class(args.val_image_dir, val_label_dir, **options)
        train_frames = {path.resolve() for sample in train_set.samples for path in sample[:2]}
        val_frames = {path.resolve() for sample in val_set.samples for path in sample[:2]}
        if train_frames & val_frames:
            raise ValueError('Training and validation folders contain the same frame paths.')
    else:
        train_set = dataset_class(args.image_dir, label_dir, split='train',
                                    val_split=args.val_split, augmentation=args.train_augmentation, **options)
        val_set = (dataset_class(args.image_dir, label_dir, split='val',
                                   val_split=args.val_split, **options) if args.val_split else None)
    return train_set, val_set


class Trainer:
    def __init__(self, args):
        self.args = args
        args.task = resolve_task(getattr(args, 'task', None), load_checkpoint(args.resume) if args.resume else None)
        self.detection = args.task == 'detection'
        self.best_names = DETECTION_BEST_NAMES if self.detection else BEST_NAMES
        self.val_fields = DETECTION_VAL_FIELDS if self.detection else VAL_FIELDS
        args.loss_version = (DETECTION_LOSS_VERSION if self.detection else
                             CUSTOM_LOSS_VERSION if args.loss == 'CUSTOM' else f'{args.loss.lower()}_v1')
        if args.resume:
            self.save_pth3 = Path(args.resume).resolve().parent
            self.save_folder3 = self.save_pth3.parent
            if self.save_folder3.parent.name == 'sessions':
                self.save_folder3 = self.save_folder3.parent.parent
                self.save_pth3 = self.save_folder3 / 'checkpoint'
        else:
            folder_name = f'{args.mode}_{args.loss}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}'
            self.save_folder3 = Path(args.work_dir) / folder_name
            self.save_pth3 = self.save_folder3 / 'checkpoint'
        self.save_pth3.mkdir(parents=True, exist_ok=True)
        self.history = RunHistory(self.save_folder3, args)
        self.session_id = self.history.session_id
        self.writer = None
        self.train_epoch_stats = self.val_epoch_stats = None
        self.validation_count = 0
        self.last_tb_image_epoch = None
        self.device = torch.device(args.device)
        checkpoint = load_checkpoint(args.resume) if args.resume else None
        args.seg_input = resolve_seg_input(args.seg_input, checkpoint)
        args.stabilization = resolve_stabilization(args.stabilization, checkpoint)
        self.train_set, self.val_set = build_datasets(args)
        if len(self.train_set) < 2:
            raise ValueError('Training needs at least two frame pairs for the head BatchNorm.')
        loader_args = dict(num_workers=args.num_workers, pin_memory=self.device.type == 'cuda',
                           worker_init_fn=init_frame_worker, collate_fn=collate_frame_boxes if self.detection else collate_frame_masks)
        batch_size = min(args.batch_size, len(self.train_set))
        # Global-pooling BatchNorm cannot train on a final batch of one sample.
        self.train_data_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                            drop_last=True, **loader_args)
        self.val_data_loader = (DataLoader(self.val_set, batch_size=args.batch_size, **loader_args)
                                if self.val_set is not None else None)
        print(f'Task: {args.task}; Train: {len(self.train_set)} pairs; validation: {len(self.val_set) if self.val_set is not None else 0}; '
              f'batch size: {batch_size}; frame_skip: {args.frame_skip}; '
              f'iterations S16/S8: {args.iters_s16}/{args.iters_s8}; stabilization: {args.stabilization}; '
              f'seg_input: {args.seg_input}; resize: {args.img_size or "native (masks match frames)"}')
        print(f'Train augmentation: {json.dumps(args.train_augmentation, sort_keys=True)}')
        self.model = ModelBuilder(seg_input=args.seg_input, task=args.task).to(self.device)
        if not args.resume:
            if args.transfer:
                transfer_model_weights(self.model, load_checkpoint(args.transfer))
            elif args.backbone_op_pretrained:
                load_weights(self.model.backbone_op, args.backbone_op_pretrained, strict=False)
        self.teacher = None
        if args.teacher:
            teacher_checkpoint = load_checkpoint(args.teacher)
            teacher_mode = resolve_seg_input(checkpoint=teacher_checkpoint)
            self.teacher = ModelBuilder(seg_input=teacher_mode).to(self.device)
            load_model_weights(self.teacher, teacher_checkpoint)
            self.teacher.requires_grad_(False).eval()
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.amp and self.device.type == 'cuda')
        self.optimizer, self.lr_scheduler = self.build_opt_lr(self.model)
        self.criterion = self.build_loss()
        self.PD_FA = None if self.detection else PD_FA(args.img_size)
        self.best_iou, self.best_nIoU, self.best_PD, self.best_FA = -1.0, -1.0, -1.0, float('inf')
        self.best_map = self.best_map50 = -1.0
        self.best_epochs = {name: None for name in self.best_names}
        self.best_checkpoints = {name: None for name in self.best_names}
        if args.resume:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])
            for key in self.best_names:
                if key in checkpoint:
                    setattr(self, key, checkpoint[key])
            self.best_epochs.update(checkpoint.get('best_epochs', {}))
            self.best_checkpoints.update(checkpoint.get('best_checkpoints', {}))
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state'])
            if self.device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        self._set_backbone_training(args.start_epoch)
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                except ImportError:
                    SummaryWriter = None
            if SummaryWriter is None:
                args.tensorboard = False
                print('TensorBoard logging disabled: install tensorboard or tensorboardX to enable it. '
                      'Training will continue with CSV/JSONL, train.log and checkpoints.')
            else:
                self.writer = SummaryWriter(log_dir=str(self.save_folder3 / 'tensorboard' / self.session_id))
                self.writer.add_text('Args', json.dumps(vars(args), indent=2))
        self.tb_sample_indices, self.tb_image_samples = select_preview_samples(
            self.val_set, args.tb_num_images, checkpoint.get('tb_image_samples', ()) if checkpoint else ())
        self.history.configure(args, checkpoint, self.tb_image_samples)
        if checkpoint:
            self._restore_best_files(checkpoint)
        self.history.set_best(self._best_records())
        if args.loss == 'CUSTOM':
            self.history.logger.info('CUSTOM loss=%s: soft intersection; threshold and --gain are ignored.',
                                     args.loss_version)
        self.history.logger.info('Train pairs=%s; validation pairs=%s; checkpoints=%s',
                                 len(self.train_set), len(self.val_set) if self.val_set is not None else 0,
                                 self.save_pth3)

    def _best_records(self):
        return {name: dict(value=json_values(getattr(self, name)) if
                           (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0) else None,
                           epoch=self.best_epochs[name], checkpoint=self.best_checkpoints[name])
                for name in self.best_names}

    def _restore_best_files(self, checkpoint):
        """Keep a rewind's best files consistent, while retaining old session weights."""
        for name in self.best_names:
            archived = self.best_checkpoints[name]
            if archived and (self.save_folder3 / archived).is_file():
                checkpoint_alias(self.save_folder3 / archived, self.save_pth3 / f'{name}.pth')
                continue
            # Legacy runs have no best epochs. Recover only verifiable metadata.
            path = self.save_pth3 / f'{name}.pth'
            if path.is_file():
                candidate = load_checkpoint(path)
                if (candidate.get(name) == getattr(self, name)
                        and candidate.get('epoch', float('inf')) <= checkpoint['epoch']):
                    self.best_epochs[name] = candidate['epoch']
                    destination = self.history.session_dir / 'checkpoint' / f'{name}.pth'
                    checkpoint_alias(path, destination)
                    self.best_checkpoints[name] = str(destination.relative_to(self.save_folder3))
                    continue
            self.best_checkpoints[name] = None

    def build_loss(self):
        if getattr(self, 'detection', False):
            return DetectionLoss()
        loss = self.args.loss
        if loss == 'CUSTOM':
            return CustomLoss()
        if loss == 'BCE':
            return criterion
        if loss == 'DICE':
            return dice_critetion
        if loss == 'SOFTIOULOSS':
            return SoftIoULoss()
        if loss == 'FOCAL':
            from external.ILNet.loss.focal_diceLoss import FocalDiceLoss_Binary
            return FocalDiceLoss_Binary()
        if loss == 'ASIMMUNIFOCAL':
            from external.ILNet.model.unified_focal_loss_pytorch import AsymmetricUnifiedFocalLoss
            return AsymmetricUnifiedFocalLoss().forward_general
        if loss == 'NAF':
            from external.ILNet.loss.NAF_losses import PSNRLoss
            return PSNRLoss(1)
        raise ValueError(f'Unsupported loss: {loss}')

    def build_opt_lr(self, model):
        args = self.args
        head_params = list(model.seg_head.parameters())
        if getattr(model, 'task', 'segmentation') == 'detection':
            head_params = list(model.seg_head.detection_feature_parameters()) + list(model.detection_head.parameters())
        params = [
            {'params': head_params, 'lr': args.learning_rate},
            {'params': model.backbone_op.parameters(), 'lr': args.learning_rate * args.backbone_layer_lr},
        ]
        optimizer_type = {'adan': Adan, 'msam': AdamW_MSAM}[args.optim]
        optimizer = optimizer_type(params, lr=args.learning_rate, weight_decay=1e-4)
        if args.lr_sch == 'cos':
            min_factor = min(1e-6 / args.learning_rate, 1.0)

            def cosine_factor(epoch):
                return min_factor + (1 - min_factor) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_factor)
        else:
            scheduler = create_lr_scheduler(optimizer, len(self.train_data_loader), args.epochs,
                                            warmup=args.warm_up_epochs > 0, warmup_epochs=args.warm_up_epochs)
        return optimizer, scheduler

    def _set_backbone_training(self, epoch):
        enabled = epoch >= self.args.epoch_training_backbone
        if enabled != getattr(self, 'backbone_training', None):
            message = (f'Epoch {epoch}: backbone {"trainable" if enabled else "frozen"}; '
                       f'LR ratio = {self.args.backbone_layer_lr}')
            if hasattr(self, 'history'):
                self.history.logger.info(message)
            else:
                print(message)
        self.backbone_training = enabled
        self.model.backbone_op.requires_grad_(enabled)
        self.model.backbone_op.train(enabled and self.model.training)
        if not enabled:
            for parameter in self.model.backbone_op.parameters():
                parameter.grad = None

    def _forward(self, model, images):
        return model(*images, iters_s16=self.args.iters_s16, iters_s8=self.args.iters_s8)

    def _loss(self, output, labels, epoch):
        main, auxiliary = self._loss_parts(output, labels)
        return main + auxiliary

    def _loss_parts(self, output, labels):
        predictions = output if isinstance(output, (list, tuple)) else [output]
        losses = [self.criterion(pred.float(), labels.float()) for pred in predictions]
        return losses[0], sum(losses[1:], losses[0].new_zeros(()))

    def _log(self, name, value, epoch):
        if self.writer is not None:
            self.writer.add_scalar(name, value, epoch)

    def training(self, epoch):
        if getattr(self, 'detection', False):
            from tool.detection_training import train_detection
            return train_detection(self, epoch)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        self._set_backbone_training(epoch)
        if self.args.optim == 'msam':
            self.optimizer.move_up_to_momentumAscent()
        totals = dict(total=0.0, main=0.0, aux=0.0, kd=0.0)
        total_samples, updates, skipped = 0, 0, 0
        stats = SegmentationStats()
        lr_values = []
        tbar = tqdm(self.train_data_loader)
        for batch in tbar:
            *images, labels = [tensor.to(self.device, non_blocking=True) for tensor in batch]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                output = self._forward(self.model, images)
                main_loss, aux_loss = self._loss_parts(output, labels)
                loss = main_loss + aux_loss
                kd_loss = loss.new_zeros(())
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_output = self._forward(self.teacher, images)
                    prediction = output[0] if isinstance(output, (list, tuple)) else output
                    teacher_output = teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output
                    temperature = self.args.T
                    kd_loss = F.binary_cross_entropy_with_logits(
                        prediction.float() / temperature, torch.sigmoid(teacher_output.float() / temperature)) * temperature**2
                    loss = self.args.alpha * loss + (1 - self.args.alpha) * kd_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}.')
            self.scaler.scale(loss).backward()
            if self.args.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            old_scale = self.scaler.get_scale()
            # Capture rates BEFORE optimizer/scheduler steps. Frozen/AMP-skipped
            # groups have no effective update and must not inflate actual LR.
            rates = [group['lr'] for group in self.optimizer.param_groups]
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # AMP overflow skips optimizer.step(); keep the schedule aligned.
            if self.scaler.get_scale() >= old_scale:
                updates += 1
                lr_values.append((rates[0], rates[1] if self.backbone_training else 0.0, labels.shape[0]))
                if self.args.lr_sch == 'ilnet':
                    self.lr_scheduler.step()
            else:
                skipped += 1
            for name, value in (('total', loss), ('main', main_loss), ('aux', aux_loss), ('kd', kd_loss)):
                totals[name] += value.item() * labels.shape[0]
            total_samples += labels.shape[0]
            prediction = output[0] if isinstance(output, (list, tuple)) else output
            stats.update(prediction, labels)
            head_lr, backbone_lr = [group['lr'] for group in self.optimizer.param_groups]
            tbar.set_description(f'Epoch {epoch}: loss={totals["total"] / total_samples:.5f}, '
                                 f'lr_head={head_lr:.3g}, lr_backbone={backbone_lr if self.backbone_training else 0:.3g}')
        if self.args.optim == 'msam':
            self.optimizer.move_back_from_momentumAscent()
        if self.args.lr_sch == 'cos' and updates:
            self.lr_scheduler.step()
        if not total_samples:
            raise RuntimeError('Training loader produced no samples.')
        values = stats.values()
        self.train_epoch_stats = dict(epoch=epoch, **{f'train_loss_{name}': value / total_samples
                                                      for name, value in totals.items()},
                                     **{f'train_{name}': values[name] for name in
                                        ('iou', 'precision', 'recall', 'target_fg_ratio', 'pred_fg_ratio')},
                                     train_samples=total_samples, backbone_training=self.backbone_training,
                                     train_seconds=time.perf_counter() - started, amp_scale=self.scaler.get_scale(),
                                     optimizer_steps=updates, optimizer_skipped_steps=skipped)
        used_samples = sum(item[2] for item in lr_values)
        for group, index in (('head', 0), ('backbone', 1)):
            self.train_epoch_stats[f'lr_{group}'] = (sum(item[index] * item[2] for item in lr_values) / used_samples
                                                    if used_samples else 0.0)
            self.train_epoch_stats[f'lr_{group}_first'] = lr_values[0][index] if lr_values else 0.0
            self.train_epoch_stats[f'lr_{group}_last'] = lr_values[-1][index] if lr_values else 0.0
        self._log('Losses/train loss', totals['total'] / total_samples, epoch)
        return totals['total'] / total_samples

    @torch.no_grad()
    def validation(self, epoch):
        if getattr(self, 'detection', False):
            from tool.detection_training import validate_detection
            return validate_detection(self, epoch)
        started = time.perf_counter()
        self.model.eval()
        self.PD_FA.reset()
        total_loss, total_samples = 0.0, 0
        stats = SegmentationStats()
        panels = {}
        selected = set(getattr(self, 'tb_sample_indices', ())) if self.writer is not None else set()
        for batch in tqdm(self.val_data_loader, desc=f'Validation {epoch}'):
            *images, labels = [tensor.to(self.device, non_blocking=True) for tensor in batch]
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                output = self._forward(self.model, images)
                output = output[0] if isinstance(output, (list, tuple)) else output
                loss = self._loss(output, labels, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite validation loss at epoch {epoch}.')
            total_loss += loss.item() * labels.shape[0]
            for offset in range(labels.shape[0]):
                index = total_samples + offset
                if index in selected:
                    panels[index] = preview_panel(images[0][offset], labels[offset], output[offset])
            total_samples += labels.shape[0]
            output, labels = output.float().cpu(), labels.cpu()
            stats.update(output, labels)
            self.PD_FA.update(output, labels)
        if not total_samples:
            raise RuntimeError('Validation loader produced no samples.')
        values = stats.values()
        iou, niou = values['iou'], values['niou']
        fa, pd = self.PD_FA.get(total_samples)
        self.val_epoch_stats = dict(epoch=epoch, val_loss=total_loss / total_samples,
                                   **{f'val_{name}': value for name, value in values.items()},
                                   val_pd=float(pd), val_fa=float(fa), val_samples=total_samples,
                                   val_seconds=time.perf_counter() - started)
        improved = []
        for name, value, better in [('best_iou', iou, iou > self.best_iou),
                                    ('best_nIoU', niou, niou > self.best_nIoU),
                                    ('best_PD', pd, pd > self.best_PD),
                                    ('best_FA', fa, fa < self.best_FA)]:
            if better:
                setattr(self, name, float(value))
                self.best_epochs[name] = epoch
                improved.append(name)
        self.validation_count += 1
        if (self.writer is not None and self.last_tb_image_epoch != epoch
                and (self.validation_count == 1 or epoch % self.args.tb_image_freq == 0 or 'best_iou' in improved)):
            for position, index in enumerate(self.tb_sample_indices):
                if index in panels:
                    self.writer.add_image(f'Validation/sample_{position:02d}_RGB_GT_pred_probability',
                                          panels[index], epoch)
            self.last_tb_image_epoch = epoch
        return improved

    def save_checkpoint(self, epoch, improved=()):
        metrics = dict(session_id=self.session_id, epoch=epoch, loss_version=self.args.loss_version,
                       **{name: None for name in getattr(self, 'val_fields', VAL_FIELDS)}, val_samples=None, val_seconds=None)
        if self.train_epoch_stats and self.train_epoch_stats['epoch'] == epoch:
            metrics.update(self.train_epoch_stats)
        if self.val_epoch_stats and self.val_epoch_stats['epoch'] == epoch:
            metrics.update(self.val_epoch_stats)
        metrics['epoch_seconds'] = (metrics.get('train_seconds', 0) + (metrics['val_seconds'] or 0))
        metrics['gpu_peak_memory_mb'] = (torch.cuda.max_memory_allocated(self.device) / 1024**2
                                         if self.device.type == 'cuda' else None)
        for name in improved:
            self.best_checkpoints[name] = str((self.history.session_dir / 'checkpoint' / f'epoch_{epoch:06d}.pth')
                                               .relative_to(self.save_folder3))
        checkpoint = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                          lr_scheduler=self.lr_scheduler.state_dict(), scaler=self.scaler.state_dict(),
                          epoch=epoch, args=json_values(vars(self.args)), rng_state=torch.get_rng_state(),
                          **{name: getattr(self, name) for name in self.best_names},
                          loss_version=self.args.loss_version, session_id=self.session_id,
                          epoch_metrics=json_values(metrics), best_epochs=dict(self.best_epochs),
                          best_checkpoints=dict(self.best_checkpoints), tb_image_samples=self.tb_image_samples)
        if self.device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        path = self.save_pth3 / 'last.pth'
        temporary_path = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
        destinations = {self.save_pth3 / f'{name}.pth' for name in improved}
        destinations.update(self.save_folder3 / self.best_checkpoints[name] for name in improved)
        save_freq = getattr(self.args, 'save_freq', 100)
        periodic_path = None
        if save_freq > 0 and epoch % save_freq == 0:
            periodic_path = self.history.session_dir / 'checkpoint' / f'epoch_{epoch:06d}.pth'
            destinations.add(periodic_path)
        for destination in destinations:
            checkpoint_alias(path, destination)
        if periodic_path is not None:
            self.history.logger.info('Periodic checkpoint at epoch %s: %s', epoch, periodic_path)
        self.history.append(metrics, self._best_records())
        for name, value in metrics.items():
            if isinstance(value, (float, int)) and name != 'epoch':
                self._log(f'Epoch/{name}', value, epoch)
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.history.close()


def main():
    args = parse_args()
    run_training(args)


def run_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Keep access to early startup logs even if dataset/model construction fails.
    trainer = Trainer.__new__(Trainer)
    try:
        trainer.__init__(args)
        for epoch in range(args.start_epoch, args.epochs + 1):
            trainer.training(epoch)
            improved = []
            if trainer.val_data_loader is not None and epoch % args.eval_freq == 0:
                improved = trainer.validation(epoch)
            trainer.save_checkpoint(epoch, improved)
        trainer.history.finish('completed')
    except KeyboardInterrupt as error:
        if hasattr(trainer, 'history'):
            trainer.history.finish('interrupted', error)
        raise
    except Exception as error:
        if hasattr(trainer, 'history'):
            trainer.history.finish('failed', error)
        raise
    finally:
        if hasattr(trainer, 'history'):
            trainer.close()
    print(f'Last checkpoint: {trainer.save_pth3 / "last.pth"}')
    if trainer.detection and trainer.best_map >= 0:
        print(f'Best AP50:95: {trainer.best_map:.5f}; best AP50: {trainer.best_map50:.5f}')
    elif trainer.best_iou >= 0:
        print(f'Best IoU: {trainer.best_iou:.5f}; best nIoU: {trainer.best_nIoU:.5f}')
    return trainer


if __name__ == '__main__':
    main()
