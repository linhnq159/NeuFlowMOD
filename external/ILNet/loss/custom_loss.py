import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, beta=2.0):
        super(CustomLoss, self).__init__()
        self.beta = 1.0/beta
        print("using custom loss")

    def forward_single(self, pred, target, threshold):
        """
        Calculates the forward pass of the custom loss function for a single prediction.

        Equation:
            intersection = pred*(pred>threshold)*target                                         # to improve the performance
            Loss = 1 - ((2*intersection + smooth)/(pred^2 + target^2 + smooth))^(1/beta)
        Args:
            pred (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The target tensor.
            threshold (float): The threshold value for binarizing the prediction. (threshold  = max_threshold*epoch/(num_epochs))

        Returns:
            torch.Tensor: The calculated loss value.
        """
                
        pred = torch.sigmoid(pred)

        temp_pred = pred > threshold
        temp_pred = temp_pred.float()

        intersection = temp_pred * target

        intersection = intersection*pred 

        smooth = 1.0
        intersection_sum = torch.sum(intersection)
        pred_sum = torch.sum(pred*pred)
        target_sum = torch.sum(target)
        loss = (2.0 * intersection_sum + smooth) / (pred_sum + target_sum + smooth)

        loss = 1 - loss**self.beta
        return loss
    
    def forward(self, pred, target, threshold=0.3):
        
        if isinstance(pred, list):
            loss = 0
            for i in range(len(pred)):
                loss += self.forward_single(pred[i], target, threshold)
            return loss
        else:
            return self.forward_single(pred, target, threshold)