import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from distillation_loss import DistillationLoss, CharbonnierLoss
from loss import flow_loss_func
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti, validate_viper

from load_model import my_load_weights, my_freeze_model
import os
from data_utils import flow_viz
import cv2


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

    def train_epoch(self, epoch):
        if self.args.distributed:
            self.student = torch.nn.parallel.DistributedDataParallel(
                self.student,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank)
            self.model_without_ddp = self.student.module
        else:
            self.model_without_ddp = self.student

        num_params = sum(p.numel() for p in self.student.parameters())
        print('Number of params:', num_params)

        scaler = torch.cuda.amp.GradScaler(enabled = self.use_fp16)
        optimizer = torch.optim.AdamW(self.model_without_ddp.parameters(), lr=self.args.lr,
                                    weight_decay=1e-4)

        start_step = 0
        
        total_steps = 0
        epoch = 0
        counter = 0

        if self.args.resume :
            state_dict = my_load_weights(self.args.resume)

            self.model_without_ddp.load_state_dict(state_dict, strict=self.args.strict_resume)

            my_freeze_model(self.student)

            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad)

            torch.save({
                'model': self.model_without_ddp.state_dict()
            }, os.path.join(self.args.checkpoint_dir, 'step_0.pth'))

        elif self.config['neuflow_checkpoint']:

            state_dict = my_load_weights(self.config['neuflow_checkpoint'])

            self.model_without_ddp.load_state_dict(state_dict, strict=self.args.strict_resume)

            my_freeze_model(self.student)

            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad)

            torch.save({
                'model': self.model_without_ddp.state_dict()
            }, os.path.join(self.args.checkpoint_dir, 'step_0.pth'))

        while total_steps < self.args.num_steps:
            """Train for one epoch."""
            self.student.train()
            self.teacher.eval()

            # manual change random seed for shuffling every epoch
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)

            for i, sample in enumerate(self.train_loader):
                print(f"epoch :{epoch}, sample {i}")
                optimizer.zero_grad()

                img1, img2, _, _ = [x.to(self.device) for x in sample]

                # img1 = img1.half()
                # img2 = img2.half()

                img1 = img1.float()
                img2 = img2.float()

                # Teacher prediction (no grad)
                with torch.no_grad():
                    teacher_flow = self.teacher(img1.contiguous(), img2.contiguous())
                    flow_gt = teacher_flow
                    # for i in range(teacher_flow.shape[0]):
                    #     flow = teacher_flow[i].permute(1,2,0).cpu().detach().numpy()
                    #     print(f"flow_teacher_{i} : {flow}")
                    #     flow = flow_viz.flow_to_image(flow)
                    #     cv2.imwrite(f"teacher_flow_{i}.png", flow)
                    
                self.model_without_ddp.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], self.device)

                with torch.cuda.amp.autocast(enabled=True):
                    flow_preds = self.student(img1, img2, iters_s16=4, iters_s8=7)
                    # for i in range(flow_preds.shape[0]):
                    #     flow = flow_preds[i].permute(1,2,0).cpu().detach().numpy()
                    #     print(f"flow_preds_{i} : {flow}")
                    #     flow = teacher_flow[i].permute(1,2,0).cpu().detach().numpy()
                    #     print(f"flow_teacher_{i} : {flow}")
                    #     flow = flow_viz.flow_to_image(flow)
                    #     cv2.imwrite(f"flow_preds_{i}.png", flow)
                    B, C, H, W = flow_gt.shape
                    # valid = torch.ones_like()
                    valid = torch.ones(B, H, W).to(self.device)
                    loss, metrics = flow_loss_func(flow_preds, flow_gt, valid, self.args.max_flow)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                bad_grad = False
                for name, param in self.student.named_parameters():
                    if not torch.all(torch.isfinite(param.grad)):
                        bad_grad = True
                    if bad_grad:
                        print(name, param.grad.mean().item())
                    # print(name, torch.max(torch.abs(param.grad)).item())

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

                scaler.step(optimizer)

                scaler.update()

                print(f"total_steps : {total_steps}, epe : {round(metrics['epe'], 3)}, mag : {round(metrics['mag'], 3)}, lr : {optimizer.param_groups[-1]['lr']}, loss : {loss}")

                total_steps += 1

                if total_steps % self.args.val_freq == 0:

                    if self.args.local_rank == 0:
                        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                        torch.save({
                            'model': self.model_without_ddp.state_dict()
                        }, checkpoint_path)

                    val_results = {}

                    if 'things' in self.args.val_dataset:
                        test_results_dict = validate_things(self.model_without_ddp, device, dstype='frames_cleanpass', validate_subset=True)
                        if self.args.local_rank == 0:
                            val_results.update(test_results_dict)

                    if 'sintel' in self.args.val_dataset:
                        test_results_dict = validate_sintel(self.model_without_ddp, self.device, dstype='final')
                        if self.args.local_rank == 0:
                            val_results.update(test_results_dict)

                    if 'kitti' in self.args.val_dataset:
                        test_results_dict = validate_kitti(self.model_without_ddp, device)
                        if self.args.local_rank == 0:
                            val_results.update(test_results_dict)

                    if 'viper' in self.args.val_dataset:
                        test_results_dict = validate_viper(self.model_without_ddp, device)
                        if self.args.local_rank == 0:
                            val_results.update(test_results_dict)
                
                    if 'U3_giam_chan' in self.args.val_dataset:
                        test_results_dict = validate_U3_giam_chan(self.model_without_ddp, self.device, self.val_dataset)
                        if self.args.local_rank == 0:
                            val_results.update(test_results_dict)

                    if self.args.local_rank == 0:
                        counter += 1

                        if counter >= 10:

                            for group in optimizer.param_groups:
                                group['lr'] *= 0.8

                            counter = 0

                        # Save validation results
                        val_file = os.path.join(self.args.checkpoint_dir, 'val_results.txt')
                        with open(val_file, 'a') as f:
                            f.write('step: %06d lr: %.6f\n' % (total_steps, optimizer.param_groups[-1]['lr']))

                            for k, v in val_results.items():
                                f.write("| %s: %.3f " % (k, v))

                            f.write('\n\n')

                    self.student.train()

            epoch += 1

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
                    student_flow, teacher_flow, flow_gt, self.task_loss_fn
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
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, task_loss, distill_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(checkpoint_dir / 'best_model.pth')
                self.logger.info(f"Saved best model at epoch {epoch}")
            
            if epoch % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(checkpoint_dir / f'model_epoch_{epoch}.pth')

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
