import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7, gamma=0.9):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.gamma = gamma  # trọng số giảm theo tầng
        self.task_loss_fn = CharbonnierLoss()

    def forward(self, student_preds, teacher_flow, valid):
        """
        Args:
            student_preds: list các tensor [B, 2, H, W]
            teacher_flow: tensor [B, 2, H, W]
            valid: tensor [B, H, W] (không dùng trong CharbonnierLoss, nhưng giữ để đồng nhất API)
        Returns:
            total_loss, task_loss, distill_loss
        """
        n_predictions = len(student_preds)
        task_loss = 0.0
        distill_loss = 0.0

        for i, student_flow in enumerate(student_preds):
            # Trọng số giảm theo tầng (giống flow_loss_func)
            i_weight = self.gamma ** (n_predictions - i - 1)

            # Task loss: student vs teacher (không dùng valid)
            i_task_loss = self.task_loss_fn(student_flow, teacher_flow)
            task_loss += i_weight * i_task_loss

            # # Distillation loss: KL divergence giữa soft targets
            # soft_student = F.log_softmax(student_flow / self.T, dim=1)  # [B, 2, H, W]
            # soft_teacher = F.softmax(teacher_flow / self.T, dim=1)      # [B, 2, H, W]
            # i_distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
            i_distill_loss = self._kl_divergence_loss(student_flow, teacher_flow)
            distill_loss += i_weight * i_distill_loss
            # distill_loss = 0

        # Total loss: (1 - alpha) * task_loss + alpha * distill_loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        # total_loss = task_loss

        return total_loss, task_loss, distill_loss

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
