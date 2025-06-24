import torch
import torch.nn as nn
import piqa

class MAELoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(MAELoss, self).__init__()
        self.name = "MAE"
        self.device = device
        self.reduction = reduction
        self.criterion = nn.L1Loss(reduction=reduction).to(self.device)

    def forward (self, img, rec):
        loss = self.criterion(img,rec)
        nb_dim = img.dim()
        if self.reduction == "none":
            return loss.mean(dim = tuple(range(1,nb_dim)))
        return loss

    def get_name(self):
        return self.name

class MSELoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(MSELoss, self).__init__()
        self.name = "MSE"
        self.device = device
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction).to(self.device)

    def forward (self, img, rec):
        loss = self.criterion(img,rec)
        nb_dim = img.dim()
        if self.reduction == "none":
            return loss.mean(dim = tuple(range(1,nb_dim)))
        return loss
    
    def get_name(self):
        return self.name


class CrossEntropyLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.name = "CrossEntropy"
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction=reduction).to(self.device)

    def forward (self, img, rec):
        return self.criterion(img,rec)
    
    def get_name(self):
        return self.name


class SSIMLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(SSIMLoss, self).__init__()
        self.device =device
        self.criterion = piqa.SSIM(reduction=reduction).to(self.device)

    def forward (self, img, rec):
        if img.shape[1]==1:
            img = img.expand(-1,3,-1,-1)
            rec = rec.repeat(1,3,1,1)
        return (1-self.criterion(img, rec))
    
    def get_name(self):
        return "SSIMLoss"
    
class MS_SSIMLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(MS_SSIMLoss, self).__init__()
        self.device =device
        self.criterion = piqa.MS_SSIM(reduction=reduction).to(self.device)

    def forward (self, img, rec):
        if img.shape[1]==1:
            img = img.expand(-1,3,-1,-1)
            rec = rec.repeat(1,3,1,1)
        return (1-self.criterion(img, rec))
    
    def get_name(self):
        return "MS_SSIMLoss"

    
class CosineDistance(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(CosineDistance, self).__init__()
        self.device = device
        self.reduction = reduction
        self.epsilon = 1e-10
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-5).to(self.device)

    def forward (self, z, p):
        # if len(z.shape)>2:
        #     z = z.reshape(z.shape[0], -1)
        # if len(p.shape)>2:
        #     p = p.reshape(p.shape[0], -1)
        loss_cos = 1-self.cos(z,p)
        if len(z.shape)>2:
            loss_cos = loss_cos.mean(dim=[1,2], keepdim=True)
        if self.reduction == "mean":
            loss_cos = loss_cos.mean()
        elif self.reduction == "sum":
            loss_cos = loss_cos.sum()   
        return loss_cos

    def get_name(self):
        return "CosineDistance"