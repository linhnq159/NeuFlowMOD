import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import torch.nn.functional as F

from distillation_loss import DistillationLoss, CharbonnierLoss
from loss import flow_loss_func
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti, validate_viper, validate_U3_giam_chan

from load_model import my_load_weights, my_freeze_model
import os
from data_utils import flow_viz
import cv2
from tqdm import tqdm
import numpy as np 

class DistillationTrainer:
    """Train NeuFlowV2 using SEA-RAFT as teacher model."""
    
    def __init__(self, student_model, teacher_model, train_loader, val_dataset, train_sampler, config, args, use_fp16):
        """
        Args:
            student_model: NeuFlowV2 model
            teacher_model: SEA-RAFT model (frozen)
            train_loader: Training data loader
            valid_loader: Validation data loader
            config: Configuration dict with hyperparameters
        """
        self.student = student_model
        self.teacher = teacher_model
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        self.config = config
        self.args = args
        self.use_fp16 = use_fp16

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student.to(self.device)
        self.teacher.to(self.device)
        
        # Loss functions
        self.task_loss_fn = CharbonnierLoss(epsilon=config.get('epsilon', 0.01))
        self.distill_loss_fn = DistillationLoss(
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 200]),
            gamma=config.get('lr_gamma', 0.1)
        )
        
        # State variables
        self.total_steps = 0
        self.epoch = 0
        self.counter = 0
        
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging."""
        logger = logging.getLogger('DistillationTrainer')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def train_epoch(self, number_epoch_training):
        loss_file = os.path.join(self.args.checkpoint_dir, 'loss.txt')

        # Wrap DDP 1 lần duy nhất
        if self.args.distributed:
            if not isinstance(self.student, torch.nn.parallel.DistributedDataParallel):
                self.student = torch.nn.parallel.DistributedDataParallel(
                    self.student,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank
                )
            self.model_without_ddp = self.student.module
        else:
            self.model_without_ddp = self.student

        num_params = sum(p.numel() for p in self.student.parameters())
        print('Number of params:', num_params)

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        # Resume checkpoint
        if self.args.resume:
            checkpoint_path = self.args.resume
            state_dict = my_load_weights(self.args.resume)

            self.model_without_ddp.load_state_dict(state_dict, strict=self.args.strict_resume)
            
            # Load optimizer state nếu có
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            
            # Load epoch/total_steps nếu có
            if 'epoch' in state_dict:
                self.epoch = state_dict['epoch']
            if 'total_steps' in state_dict:
                self.total_steps = state_dict['total_steps']
            
            my_freeze_model(self.student)

            torch.save({
                'model': self.model_without_ddp.state_dict(),
                'total_steps': self.total_steps,
                'epoch': self.epoch,
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.args.checkpoint_dir, 'step_0.pth'))

        while self.total_steps < self.args.num_steps and self.epoch < number_epoch_training:
            """Train for one epoch."""
            self.student.train()
            self.teacher.eval()
            list_loss = []
            list_epe = []

            # manual change random seed for shuffling every epoch
            if self.args.distributed:
                self.train_sampler.set_epoch(self.epoch)

            tbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", dynamic_ncols=True, disable=self.args.local_rank != 0)
            # tbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=self.args.local_rank != 0)
            # tbar = tqdm(self.train_loader, disable=self.args.local_rank != 0)
            for i, sample in enumerate(tbar):

            # for i, sample in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                img1, img2, _, _ = [x.to(self.device) for x in sample]

                # img1 = img1.half()
                # img2 = img2.half()

                # img1 = img1.float()
                # img2 = img2.float()

                # Teacher prediction (no grad)
                with torch.no_grad():
                    teacher_flow = self.teacher(img1.contiguous(), img2.contiguous())
                    # for i in range(teacher_flow.shape[0]):
                    #     flow = teacher_flow[i].permute(1,2,0).cpu().detach().numpy()
                    #     print(f"flow_teacher_{i} : {flow}")
                    #     flow = flow_viz.flow_to_image(flow)
                    #     cv2.imwrite(f"teacher_flow_{i}.png", flow)
                    
                    # print("teacher_flow : ", teacher_flow)
                    # Tạo valid từ teacher flow
                    mag = torch.sum(teacher_flow ** 2, dim=1).sqrt()
                    valid = (mag < self.args.max_flow).float()  # [B, H, W]
                self.model_without_ddp.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], self.device, amp = True)

                with torch.cuda.amp.autocast(enabled=True):
                    flow_preds = self.student(img1, img2, iters_s16=4, iters_s8=7)
                    # print("flow_preds : ", flow_preds[-1])
                    # for i in range((flow_preds[-1]).shape[0]):
                    #     flow = flow_preds[-1][i].permute(1,2,0).cpu().detach().numpy()
                    #     print(f"flow_preds_{i} : {flow}")
                    #     # flow = teacher_flow[i].permute(1,2,0).cpu().detach().numpy()
                    #     # print(f"flow_teacher_{i} : {flow}")
                    #     flow = flow_viz.flow_to_image(flow)
                    #     cv2.imwrite(f"flow_preds_{i}.png", flow)
                    
                    loss, task_loss, distill_loss = self.distill_loss_fn(
                        flow_preds, teacher_flow, valid
                    )

                    # B, C, H, W = teacher_flow.shape
                    # valid = torch.ones_like()
                    # valid = torch.ones(B, H, W).to(self.device)
                    _, metrics = flow_loss_func(flow_preds, teacher_flow, valid, self.args.max_flow)
                    list_loss.append(loss.item())
                    list_epe.append(metrics['epe'])

                # x = input()

                with open(loss_file, 'a') as f1:
                    f1.write('epoch %3d, step: %06d, lr: %.6f, loss: %.6f\n' % (self.epoch, self.total_steps, self.optimizer.param_groups[-1]['lr'], loss))

                # print("task_loss : ", task_loss)
                # print("distill_loss : ", distill_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)

                bad_grad = False
                for name, param in self.student.named_parameters():
                    if not torch.all(torch.isfinite(param.grad)):
                        bad_grad = True
                    if bad_grad:
                        print(name, param.grad.mean().item())
                    # print(name, torch.max(torch.abs(param.grad)).item())

                # bad_grad = False
                # for name, param in self.student.named_parameters():
                #     if param.grad is None:
                #         continue

                #     if not torch.all(torch.isfinite(param.grad)):
                #         bad_grad = True
                #         # print(name, param.grad.mean().item())
                #     # print(name, torch.max(torch.abs(param.grad)).item())

                # # print(f"epe_mean : {round(metrics['epe'], 3)}, loss : {loss}")

                # if bad_grad:
                #     self.logger.warning("Skipping update due to non-finite gradients.")
                #     self.optimizer.zero_grad(set_to_none = True)
                #     scaler.update()
                #     continue

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                self.total_steps += 1

                # # Tính EPE từ teacher_flow (để logging)
                # epe = torch.sum((flow_preds[-1] - teacher_flow) ** 2, dim=1).sqrt()
                # print("epe : ", epe)
                # print("epe.shape : ", epe.shape)
                # epe = epe.view(-1)[valid.view(-1).bool()]
                # print("epe after view: ", epe)
                # if epe.numel() > 0:
                #     epe_mean = epe.mean().item()
                # else:
                #     epe_mean = 0.0
                
                # mag_mean = mag.mean().item()

                # print(f"epe_mean : {round(metrics['epe'], 3)}, loss : {loss}")

                tbar.set_description(f"Epoch:{self.epoch}, step:{self.total_steps}, lr:{self.optimizer.param_groups[-1]['lr']:.6f}, epe:{round(np.mean(list_epe), 3)}, loss:{round(np.mean(list_loss), 3)}")

            if self.total_steps % 1 == 0 and self.epoch % 5 == 0:
                if self.args.local_rank == 0:
                    checkpoint_path = os.path.join(self.args.checkpoint_dir, 'epoch_%03d_step_%06d.pth' % (self.epoch, self.total_steps))
                    torch.save({
                        'model': self.model_without_ddp.state_dict(),
                        'total_steps': self.total_steps,
                        'epoch': self.epoch,
                        'optimizer': self.optimizer.state_dict(),
                    }, checkpoint_path)

                val_results = {}

                if 'things' in self.args.val_dataset:
                    test_results_dict = validate_things(self.model_without_ddp, self.device, dstype='frames_cleanpass', validate_subset=True)
                    if self.args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'sintel' in self.args.val_dataset:
                    test_results_dict = validate_sintel(self.model_without_ddp, self.device, dstype='final')
                    if self.args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'kitti' in self.args.val_dataset:
                    test_results_dict = validate_kitti(self.model_without_ddp, self.device)
                    if self.args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'viper' in self.args.val_dataset:
                    test_results_dict = validate_viper(self.model_without_ddp, self.device)
                    if self.args.local_rank == 0:
                        val_results.update(test_results_dict)
                
                if 'U3_giam_chan' in self.args.val_dataset:
                    test_results_dict = validate_U3_giam_chan(model=self.model_without_ddp, teacher_model=self.teacher, device=self.device, val_dataset=self.val_dataset, dstype='dot1_T32026')
                    if self.args.local_rank == 0:
                        val_results.update(test_results_dict)

                
                self.counter += 1

                if self.counter >= 10:
                    for group in self.optimizer.param_groups:
                        group['lr'] *= 0.8
                    self.counter = 0

                if self.args.local_rank == 0:
                    # Save validation results
                    val_file = os.path.join(self.args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('epoch %3d, step: %06d lr: %.6f\n' % (self.epoch, self.total_steps, self.optimizer.param_groups[-1]['lr']))
                        for k, v in val_results.items():
                            f.write("| %s: %.3f " % (k, v))
                        f.write('\n\n')

                self.student.train()
            
            self.epoch += 1

    def validate(self, epoch):
        """Validate on validation set."""
        self.student.eval()
        self.teacher.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                flow_gt = batch['flow'].to(self.device)
                
                student_flow = self.student(img1.contiguous(), img2.contiguous())
                teacher_flow = self.teacher(img1.contiguous(), img2.contiguous())
                
                loss, _, _ = self.distill_loss_fn(
                    student_flow, teacher_flow, torch.ones_like(flow_gt[:, 0, :, :]), self.task_loss_fn
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
        
        return avg_loss

    def train(self, num_epochs, checkpoint_dir='./checkpoints'):
        """Train for specified number of epochs."""
        checkpoint_dir = Path(checkpoint_dir)
        # checkpoint_dir.mkdir(exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok = True)
        
        best_val_loss = float('inf')
        
        # Train
        self.train_epoch(num_epochs)
        
        # Validate
        val_loss = self.validate(self.epoch)
        
        # Update learning rate
        self.scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self._save_checkpoint(checkpoint_dir / 'best_model.pth')
            self.logger.info(f"Saved best model at epoch {self.epoch}")
        
        if self.epoch % self.config.get('save_interval', 10) == 0:
            self._save_checkpoint(checkpoint_dir / f'model_epoch_{self.epoch}.pth')

    def _save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")