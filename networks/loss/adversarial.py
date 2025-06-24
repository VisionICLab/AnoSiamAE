from torch import nn
import torch

class BinaryAdversarialLoss(nn.Module):
    """
        Inspired from monai PatchAdversarialLoss implementation
        https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/adversarial_loss.py
    """
    def __init__(self, device="cuda", reduction = "mean"):
        super(BinaryAdversarialLoss, self).__init__()
        self.device = device
        self.reduction = reduction
        self.criterion = nn.BCELoss(reduction=reduction).to(device)
        self.real_labels = torch.tensor(1.0).to(device)
        self.fake_labels = torch.tensor(0.0).to(device)
        
    def forward (self, outD, is_real, for_disc):
        if not for_disc and not is_real:
            is_real = True
        target = self.real_labels if is_real else self.fake_labels
        target = target.expand_as(outD)
        loss = self.criterion(outD, target)
        return loss        

    def get_name(self):
        return "BinaryAdversarialLoss"

class PatchAdversarialLoss(nn.Module):
    """
        Inspired from monai PatchAdversarialLoss implementation
        https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/adversarial_loss.py
    """
    def __init__(self, device="cuda", reduction = "mean"):
        super(PatchAdversarialLoss, self).__init__()
        self.device = device
        self.reduction = reduction
        self.criterion = nn.BCELoss(reduction=reduction).to(device)
        self.real_labels = torch.tensor(1.0).to(device)
        self.fake_labels = torch.tensor(0.0).to(device)
        self.sig = nn.Sigmoid()

    def forward (self, outPatchD, is_real, for_disc):
        d_logits = self.sig(outPatchD)
        if not for_disc and not is_real:
            is_real = True
        if type(d_logits) is not list:
            d_logits = [d_logits]
        target = []
        for output in d_logits:
            label = self.real_labels if is_real else self.fake_labels
            label_tensor = label.expand_as(output)
            target.append(label_tensor)
        loss = []
        for targ, output in zip(target,d_logits):
            loss.append(self.criterion(output, targ))
        if self.reduction == "mean":
            loss = torch.mean(torch.stack(loss))
        elif self.reduction == "sum":
            loss = torch.sum(torch.stack(loss))
        return loss        

    def get_name(self):
        return "PatchAdversarialLoss"
