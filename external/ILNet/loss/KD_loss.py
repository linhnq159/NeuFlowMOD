import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional  as F

class CosinSimilarity_Loss(nn.Module):
    def __init__(self, dim = 2):
        super(CosinSimilarity_Loss,self).__init__()
        self.dim = dim

    def forward_single(self,input1, input2):
        return F.cosine_similarity(input1, input2, dim=self.dim).mean()
    
    def forward(self,inputs1, inputs2, sigmoid = True):
        total_loss = 0.0

        if sigmoid:
            inputs1 = torch.sigmoid(inputs1)
            inputs2 = torch.sigmoid(inputs2)
        
        assert inputs1.size() == inputs2.size(), 'inputs1 {} & inputs2 {} shape do not match'.format(inputs1.size(), inputs2.size())
        
        if isinstance(inputs1, list):
            for i in range(len(inputs1)):
                total_loss += self.forward_single(inputs1[i], inputs2[i])
        else:
            total_loss = self.forward_single(inputs1, inputs2)
        return total_loss