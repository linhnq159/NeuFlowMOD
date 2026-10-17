"""Train frame-pair motion segmentation from images and current-frame masks."""

import json
import math
import random
import sys
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

from data_utils.frame_mask_dataset import FrameMaskDataset
from data_utils.frame_pair import init_frame_worker
from tool.checkpoints import (load_checkpoint, load_weights, load_model_weights,
                              resolve_stabilization, resolve_seg_input, transfer_model_weights)
from NeuFlow.config import SEG_INPUT_MODES
from NeuFlow.model_builder import ModelBuilder
from external.ILNet.utils.Adan import Adan
from external.ILNet.optimizer.adamW_msam import AdamW_MSAM
from external.ILNet.utils.lr_scheduler import create_lr_scheduler
from external.ILNet.model.loss import SoftIoULoss, criterion, dice_critetion
from external.ILNet.model.metrics import IoUMetric, nIoUMetric, PD_FA
from external.ILNet.loss.custom_loss import CustomLoss


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--work_dir', default='work_seg/checkpoint')
    parser.add_argument('--img_size', nargs=2, type=int, default=[960, 544], metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--warm_up_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optim', choices=['adan', 'msam'], default='adan')
    parser.add_argument('--lr_sch', choices=['ilnet', 'cos'], default='ilnet')
    parser.add_argument('--seg_input', choices=SEG_INPUT_MODES, default=None,
                        help='Head input; new runs use NeuFlow/config.py, resume uses checkpoint')
    parser.add_argument('--data_path', help='Dataset root with images/ and masks/')
    parser.add_argument('--image_dir', '--train_image_dir', dest='image_dir', help='Training image folder')
    parser.add_argument('--mask_dir', '--train_mask_dir', dest='mask_dir', help='Training ground-truth mask folder')
    parser.add_argument('--val_image_dir', help='Optional separate validation image folder')
    parser.add_argument('--val_mask_dir', help='Optional separate validation mask folder')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Last fraction of each sequence reserved for validation; 0 disables it')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Distance in sorted frames: inputs (t, t-frame_skip), target mask at t')
    parser.add_argument('--iters_s16', type=int, default=1, help='Number of S16 refinement iterations')
    parser.add_argument('--iters_s8', type=int, default=8, help='Number of S8 refinement iterations')
    parser.add_argument('--stabilization', choices=['none', 'homography'], default=None,
                        help='Past-to-current alignment; new runs default to homography, resume uses checkpoint')
    parser.add_argument('--mode', default='L')
    parser.add_argument('--loss', type=str.upper, default='CUSTOM',
                        choices=['BCE', 'SOFTIOULOSS', 'DICE', 'FOCAL', 'CUSTOM', 'ASIMMUNIFOCAL', 'NAF'])
    parser.add_argument('--resume', default='', help='Resume model, optimizer, scheduler and epoch')
    parser.add_argument('--transfer', default='', help='Initialize model weights from a checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--amp', action=BooleanOptionalAction, default=True)
    parser.add_argument('--tensorboard', action=BooleanOptionalAction, default=True)
    parser.add_argument('--teacher', default='', help='Optional ModelBuilder checkpoint for distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Supervised loss weight when using a teacher')
    parser.add_argument('--T', type=float, default=10, help='Distillation temperature')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--gain', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=0.0, help='Gradient norm limit; 0 disables clipping')
    parser.add_argument('--seed', type=int, default=1211)
    parser.add_argument('--backbone_op_pretrained', default=str(ROOT / 'neuflow_sintel.pth'),
                        help='Pretrained NeuFlow weights; pass an empty string to initialize randomly')
    parser.add_argument('--epoch_training_backbone', type=int, default=300,
                        help='First epoch training the backbone: epochs 1..299 are frozen by default')
    parser.add_argument('--backbone_layer_lr', type=float, default=0.1,
                        help='Backbone LR / head LR throughout training')
    args = parser.parse_args(argv)
    if args.resume and args.transfer:
        parser.error('Choose either --resume or --transfer, not both.')
    if args.batch_size < 2:
        parser.error('--batch_size must be >= 2 because the head has BatchNorm after global pooling.')
    if args.epochs < 1 or args.start_epoch < 1 or args.start_epoch > args.epochs:
        parser.error('Require 1 <= start_epoch <= epochs.')
    if args.frame_skip < 1 or args.eval_freq < 1 or args.num_workers < 0:
        parser.error('frame_skip/eval_freq must be >= 1 and num_workers must be >= 0.')
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
    if min(args.img_size) < 32 or any(size % 16 for size in args.img_size):
        parser.error('--img_size WIDTH HEIGHT must be >= 32 and divisible by 16.')
    if bool(args.val_image_dir) != bool(args.val_mask_dir):
        parser.error('Provide both --val_image_dir and --val_mask_dir.')
    if args.data_path:
        args.image_dir = args.image_dir or str(Path(args.data_path) / 'images')
        args.mask_dir = args.mask_dir or str(Path(args.data_path) / 'masks')
    if not args.image_dir or not args.mask_dir:
        parser.error('Provide --image_dir and --mask_dir, or --data_path with images/ and masks/.')
    return args


def build_datasets(args):
    args.seg_input = resolve_seg_input(args.seg_input, load_checkpoint(args.resume) if args.resume else None)
    if args.stabilization is None:
        args.stabilization = resolve_stabilization(None, load_checkpoint(args.resume) if args.resume else None)
    options = dict(img_size=args.img_size, frame_skip=args.frame_skip, stabilization=args.stabilization)
    if args.val_image_dir:
        train_set = FrameMaskDataset(args.image_dir, args.mask_dir, **options)
        val_set = FrameMaskDataset(args.val_image_dir, args.val_mask_dir, **options)
        train_frames = {path.resolve() for sample in train_set.samples for path in sample[:2]}
        val_frames = {path.resolve() for sample in val_set.samples for path in sample[:2]}
        if train_frames & val_frames:
            raise ValueError('Training and validation folders contain the same frame paths.')
    else:
        train_set = FrameMaskDataset(args.image_dir, args.mask_dir, split='train',
                                    val_split=args.val_split, **options)
        val_set = (FrameMaskDataset(args.image_dir, args.mask_dir, split='val',
                                   val_split=args.val_split, **options) if args.val_split else None)
    return train_set, val_set


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        checkpoint = load_checkpoint(args.resume) if args.resume else None
        args.seg_input = resolve_seg_input(args.seg_input, checkpoint)
        args.stabilization = resolve_stabilization(args.stabilization, checkpoint)
        self.train_set, self.val_set = build_datasets(args)
        if len(self.train_set) < 2:
            raise ValueError('Training needs at least two frame pairs for the head BatchNorm.')
        loader_args = dict(num_workers=args.num_workers, pin_memory=self.device.type == 'cuda',
                           worker_init_fn=init_frame_worker)
        batch_size = min(args.batch_size, len(self.train_set))
        # Global-pooling BatchNorm cannot train on a final batch of one sample.
        self.train_data_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                            drop_last=True, **loader_args)
        self.val_data_loader = (DataLoader(self.val_set, batch_size=args.batch_size, **loader_args)
                                if self.val_set is not None else None)
        print(f'Train: {len(self.train_set)} pairs; validation: {len(self.val_set) if self.val_set is not None else 0}; '
              f'batch size: {batch_size}; frame_skip: {args.frame_skip}; '
              f'iterations S16/S8: {args.iters_s16}/{args.iters_s8}; stabilization: {args.stabilization}; '
              f'seg_input: {args.seg_input}')
        self.model = ModelBuilder(seg_input=args.seg_input).to(self.device)
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
        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(args.img_size)
        self.best_iou, self.best_nIoU, self.best_PD, self.best_FA = -1.0, -1.0, -1.0, float('inf')
        if args.resume:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])
            for key in ('best_iou', 'best_nIoU', 'best_PD', 'best_FA'):
                if key in checkpoint:
                    setattr(self, key, checkpoint[key])
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state'])
            if self.device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        self._set_backbone_training(args.start_epoch)
        if args.resume:
            self.save_pth3 = Path(args.resume).resolve().parent
            self.save_folder3 = self.save_pth3.parent
        else:
            folder_name = f'{args.mode}_{args.loss}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}'
            self.save_folder3 = Path(args.work_dir) / folder_name
            self.save_pth3 = self.save_folder3 / 'checkpoint'
        self.save_pth3.mkdir(parents=True, exist_ok=True)
        (self.save_folder3 / 'config.json').write_text(json.dumps(vars(args), indent=2))
        self.writer = None
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.save_folder3))
            self.writer.add_text('Args', json.dumps(vars(args), indent=2))
        print(f'Checkpoints: {self.save_pth3}')

    def build_loss(self):
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
        params = [
            {'params': model.seg_head.parameters(), 'lr': args.learning_rate},
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
            print(f'Epoch {epoch}: backbone {"trainable" if enabled else "frozen"}; '
                  f'LR ratio = {self.args.backbone_layer_lr}')
        self.backbone_training = enabled
        self.model.backbone_op.requires_grad_(enabled)
        self.model.backbone_op.train(enabled and self.model.training)
        if not enabled:
            for parameter in self.model.backbone_op.parameters():
                parameter.grad = None

    def _forward(self, model, images):
        return model(*images, iters_s16=self.args.iters_s16, iters_s8=self.args.iters_s8)

    def _loss(self, output, labels, epoch):
        predictions = output if isinstance(output, (list, tuple)) else [output]
        if self.args.loss == 'CUSTOM':
            threshold = min(self.args.gain * epoch / max(self.args.epochs - 50, 1), 0.2)
            return sum(self.criterion(pred.float(), labels.float(), threshold) for pred in predictions)
        return sum(self.criterion(pred.float(), labels.float()) for pred in predictions)

    def _log(self, name, value, epoch):
        if self.writer is not None:
            self.writer.add_scalar(name, value, epoch)

    def training(self, epoch):
        self.model.train()
        self._set_backbone_training(epoch)
        if self.args.optim == 'msam':
            self.optimizer.move_up_to_momentumAscent()
        total_loss, total_samples, updates = 0.0, 0, 0
        tbar = tqdm(self.train_data_loader)
        for batch in tbar:
            *images, labels = [tensor.to(self.device, non_blocking=True) for tensor in batch]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                output = self._forward(self.model, images)
                loss = self._loss(output, labels, epoch)
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_output = self._forward(self.teacher, images)
                    prediction = output[0] if isinstance(output, (list, tuple)) else output
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
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # AMP overflow skips optimizer.step(); keep the schedule aligned.
            if self.scaler.get_scale() >= old_scale:
                updates += 1
                if self.args.lr_sch == 'ilnet':
                    self.lr_scheduler.step()
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            head_lr, backbone_lr = [group['lr'] for group in self.optimizer.param_groups]
            tbar.set_description(f'Epoch {epoch}: loss={total_loss / total_samples:.5f}, '
                                 f'lr_head={head_lr:.3g}, lr_backbone={backbone_lr if self.backbone_training else 0:.3g}')
        if self.args.optim == 'msam':
            self.optimizer.move_back_from_momentumAscent()
        if self.args.lr_sch == 'cos' and updates:
            self.lr_scheduler.step()
        self._log('Losses/train loss', total_loss / total_samples, epoch)
        self._log('Learning rate/head', self.optimizer.param_groups[0]['lr'], epoch)
        self._log('Learning rate/backbone', self.optimizer.param_groups[1]['lr'] if self.backbone_training else 0, epoch)
        return total_loss / total_samples

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        total_loss, total_samples = 0.0, 0
        for batch in tqdm(self.val_data_loader, desc=f'Validation {epoch}'):
            *images, labels = [tensor.to(self.device, non_blocking=True) for tensor in batch]
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                output = self._forward(self.model, images)
                loss = self._loss(output, labels, epoch)
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            output, labels = output.float().cpu(), labels.cpu()
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)
        _, iou = self.iou_metric.get()
        _, niou = self.nIoU_metric.get()
        fa, pd = self.PD_FA.get(total_samples)
        metrics = dict(loss=total_loss / total_samples, IoU=float(iou), nIoU=float(niou), Pd=float(pd), Fa=float(fa))
        print(f'Validation {epoch}: {metrics}')
        for name, value in metrics.items():
            self._log(f'Eval/{name}', value, epoch)
        improved = []
        for name, value, better in [('best_iou', iou, iou > self.best_iou),
                                    ('best_nIoU', niou, niou > self.best_nIoU),
                                    ('best_PD', pd, pd > self.best_PD),
                                    ('best_FA', fa, fa < self.best_FA)]:
            if better:
                setattr(self, name, float(value))
                improved.append(name)
        return improved

    def save_checkpoint(self, epoch, improved=()):
        checkpoint = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                          lr_scheduler=self.lr_scheduler.state_dict(), scaler=self.scaler.state_dict(),
                          epoch=epoch, args=vars(self.args), rng_state=torch.get_rng_state(),
                          best_iou=self.best_iou, best_nIoU=self.best_nIoU,
                          best_PD=self.best_PD, best_FA=self.best_FA)
        if self.device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        for name in ('last', *improved):
            path = self.save_pth3 / f'{name}.pth'
            temporary_path = path.with_suffix('.tmp')
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    try:
        for epoch in range(args.start_epoch, args.epochs + 1):
            trainer.training(epoch)
            improved = []
            if trainer.val_data_loader is not None and epoch % args.eval_freq == 0:
                improved = trainer.validation(epoch)
            trainer.save_checkpoint(epoch, improved)
    finally:
        if trainer.writer is not None:
            trainer.writer.close()
    print(f'Last checkpoint: {trainer.save_pth3 / "last.pth"}')
    if trainer.best_iou >= 0:
        print(f'Best IoU: {trainer.best_iou:.5f}; best nIoU: {trainer.best_nIoU:.5f}')


if __name__ == '__main__':
    main()
