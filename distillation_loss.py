import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combines task loss with knowledge distillation loss.
    Student (NeuFlowV2) learns from teacher (SEA-RAFT) predictions.
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for task loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_output, teacher_output, ground_truth, task_loss_fn):
        """
        Args:
            student_output: NeuFlowV2 predictions (B, 2, H, W)
            teacher_output: SEA-RAFT predictions (B, 2, H, W)
            ground_truth: Ground truth flow (B, 2, H, W)
            task_loss_fn: Function to compute task loss (e.g., EPE, Charbonnier)
        
        Returns:
            Total loss
        """
        # Task loss (student predicting ground truth)
        task_loss = task_loss_fn(student_output, ground_truth)
        
        # KL divergence loss (distillation)
        kl_loss = self._kl_divergence_loss(student_output, teacher_output)
        
        # Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * kl_loss
        
        return total_loss, task_loss, kl_loss

    def _kl_divergence_loss(self, student_flow, teacher_flow):
        """
        Compute KL divergence between normalized student and teacher flows.
        """
        # Normalize flows
        student_norm = F.normalize(student_flow, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flow, p=2, dim=1)
        
        # Compute L2 distance between flows
        flow_diff = torch.norm(student_norm - teacher_norm, p=2, dim=1)
        
        return flow_diff.mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier loss for optical flow."""
    def __init__(self, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted flow (B, 2, H, W)
            target: Target flow (B, 2, H, W)
        """
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))
        return loss
