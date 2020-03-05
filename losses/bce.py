import torch
import torch.nn as nn

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        target = target.type_as(output)
        return self.loss(output, target.unsqueeze(1))

class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, beta, **kwargs):
        if isinstance(beta, (float,int)):
            self.beta = torch.Tensor([beta])
        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)
        super().__init__(pos_weight=self.beta, **kwargs)