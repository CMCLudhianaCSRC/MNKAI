import torch, torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__(); self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-3)

def temperature_scale(logits, labels, max_iter=1000, lr=0.01):
    ts = TemperatureScaler().to(logits.device)
    opt = torch.optim.LBFGS([ts.temperature], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(ts(logits), labels); loss.backward(); return loss
    opt.step(closure)
    return ts
