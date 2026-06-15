import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from distillation_loss import DistillationLoss, CharbonnierLoss


class DistillationTrainer:
    """Train NeuFlowV2 using SEA-RAFT as teacher model."""
    
    def __init__(self, student_model, teacher_model, train_loader, val_loader, config):
        """
        Args:
            student_model: NeuFlowV2 model
            teacher_model: SEA-RAFT model (frozen)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dict with hyperparameters
        """
        self.student = student_model
        self.teacher = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
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
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_distill_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            flow_gt = batch['flow'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Student prediction
            student_flow = self.student(img1, img2)
            
            # Teacher prediction (no grad)
            with torch.no_grad():
                teacher_flow = self.teacher(img1, img2)
            
            # Compute distillation loss
            total, task_loss, distill_loss = self.distill_loss_fn(
                student_flow, teacher_flow, flow_gt, self.task_loss_fn
            )
            
            # Backward pass
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total.item()
            total_task_loss += task_loss.item()
            total_distill_loss += distill_loss.item()
            
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {total.item():.4f} "
                    f"(Task: {task_loss.item():.4f}, Distill: {distill_loss.item():.4f})"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        avg_task_loss = total_task_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        
        return avg_loss, avg_task_loss, avg_distill_loss

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
                
                student_flow = self.student(img1, img2)
                teacher_flow = self.teacher(img1, img2)
                
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
        checkpoint_dir.mkdir(exist_ok=True)
        
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
