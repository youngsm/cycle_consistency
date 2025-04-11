import torch
import torch.nn as nn

class PhotoelectronSparsityLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, weight: float = 0.01):
        super().__init__()

        self.alpha = alpha
        self.weight = weight
    def forward(self, model_output):
        return self.weight * model_output['pred_pe_weighted'].sum(dim=-1).pow(self.alpha).mean()

class ConfidenceSparsityLoss(nn.Module):
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, model_output):
        return self.weight * model_output['pred_c'].norm(p=1, dim=-1).mean()