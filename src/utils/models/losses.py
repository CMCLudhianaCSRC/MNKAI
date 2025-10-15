import torch, torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, reduction='mean'):
        super().__init__(); self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1-pt)**self.gamma) * logp.gather(1, target.unsqueeze(1)).squeeze(1)
        return loss.mean() if self.reduction=='mean' else loss.sum()

def effective_num_weights(labels, num_classes, beta=0.9999):
    counts = np.bincount(labels, minlength=num_classes)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)
