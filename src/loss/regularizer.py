import torch
import torch.nn as nn

class KLLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(KLLoss, self).__init__()
        self.device = device
        self.reduction = reduction
      
    def forward (self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(mu.shape[0], -1)
        kl_loss = 0.5*torch.sum(mu.pow(2)+torch.exp(logvar)-logvar-1, dim=1)
        if self.reduction == "mean":
            kl_loss = kl_loss.mean()
        elif self.reduction == "sum":
            kl_loss = kl_loss.sum()   
        return kl_loss

    def get_name(self):
        return "KLLoss"