import torch.nn as nn
import torch
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, reduction:str = "mean"):
        super().__init__()

        self.reduction = reduction
        self.criterion = nn.BCELoss(reduction=reduction)

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return self.criterion(prediction.float(), target.float())

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction:str = "mean"):
        super().__init__()

        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return self.criterion(prediction, target)

class DiceLoss(nn.Module):
    def __init__(self, reduction:str = "mean", epsilon:float = 1E-06):
        super().__init__()

        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if prediction.shape[1]==1:
            if len(target.shape)==3:
                target = target.unsqueeze(1)
            intersection = torch.sum(prediction*target, dim=[-1,-2])
            union = torch.sum(prediction*prediction, dim=[-1,-2]) + torch.sum(target, dim=[-1,-2])
            dice = 1-torch.mean((2.0*intersection + self.epsilon)/(union + self.epsilon), dim=1)
        else:
            target = F.one_hot(target, num_classes = prediction.shape[1]).permute(0,3,1,2)
            intersection = torch.sum(prediction*target, dim=[-1,-2])
            union = torch.sum(prediction*prediction, dim=[-1,-2]) + torch.sum(target, dim=[-1,-2])
            dice = 1-torch.mean((2.0*intersection + self.epsilon)/(union + self.epsilon), dim=1)
        
        
        if self.reduction=="mean":
            return dice.mean()
        elif self.reduction=="sum":
            return dice.sum()
        return dice
    
    
class JaccardLoss(nn.Module):
    def __init__(self, reduction:str = "mean", epsilon:float = 1E-06):
        super().__init__()

        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if len(target.shape)==3:
            target = target.unsqueeze(1)

        tp = (prediction*target).sum(dim=[-1,-2])
        fn = ((1-prediction)*target).sum(dim=[-1,-2])
        fp = (prediction*(1-target)).sum(dim=[-1,-2])

        jaccard = 1-torch.mean((tp + self.epsilon)/(tp + fn + fp + self.epsilon),dim=1)
        if self.reduction == "mean":
            return jaccard.mean()
        if self.reduction == "mean":
            return jaccard.sum()

        return jaccard

class TverskyLoss(nn.Module):
    def __init__(self, reduction:str = "mean", alpha:float = 0.5, epsilon:float = 1.0E-6):
        super().__init__()
        assert alpha>0 and alpha<1.0, "Error. Alpha should be between 0 and 1"
        self.reduction = reduction
        self.alpha = alpha # Alpha==0 -> emphasis on FP || Alpha==1 -> emphasis on FN
        self.epsilon = epsilon

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if len(target.shape)==3:
            target = target.unsqueeze(1)

        tp = (prediction*target).sum(dim=[-1,-2])
        fn = ((1-prediction)*target).sum(dim=[-1,-2])
        fp = (prediction*(1-target)).sum(dim=[-1,-2])

        tversky = torch.mean(1-(tp + self.epsilon)/(tp + self.alpha*fn + (1-self.alpha)*fp + self.epsilon), dim=1)
        if self.reduction == "mean":
            return tversky.mean()
        if self.reduction == "mean":
            return tversky.sum()

        return tversky
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, reduction:str = "mean", alpha:float = 0.5, gamma:float=1.0, epsilon:float = 1.0E-6):
        super().__init__()
        assert alpha>0 and alpha<1.0, "Error. Alpha should be between 0 and 1"
        self.reduction = reduction
        self.alpha = alpha # Alpha==0 -> emphasis on FP || Alpha==1 -> emphasis on FN
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if len(target.shape)==3:
            target =target.unsqueeze(1)

        tp = (prediction*target).sum(dim=[-1,-2])
        fn = ((1-prediction)*target).sum(dim=[-1,-2])
        fp = (prediction*(1-target)).sum(dim=[-1,-2])

        ftversky = torch.mean((1-(tp + self.epsilon)/(tp + self.alpha*fn + (1-self.alpha)*fp + self.epsilon)).pow(self.gamma), dim=1)
        if self.reduction == "mean":
            return ftversky.mean()
        if self.reduction == "mean":
            return ftversky.sum()
        
        return ftversky
    
class FocalLoss(nn.Module):
    def __init__(self, reduction:str = "mean", alpha:float = 0.25, gamma:float=2.0):
        super().__init__()
        assert alpha>0 and alpha<1.0, "Error. Alpha should be between 0 and 1"
        self.reduction = reduction
        self.alpha = alpha # Alpha==0 -> emphasis on FP || Alpha==1 -> emphasis on FN
        self.gamma = gamma
        self.criterion = nn.BCELoss(reduction="none")

    def forward(self, logits:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        one_hot_key = F.one_hot(target.squeeze(1).long()).permute(0,3,1,2) # [B,NC,**]
        log_pt = logits.log_softmax(1) # [B,NC,**]
        loss = -torch.pow(1.0-log_pt.exp(), self.gamma)*log_pt*one_hot_key
        alpha = torch.cat([((1.0-self.alpha)*one_hot_key[:,0]).unsqueeze(1),self.alpha*one_hot_key[:,1:]], dim=1) # [B,NC,**]
        loss = torch.sum(alpha*loss,1, keepdim=True)
        if self.reduction=="mean":
            loss = loss.mean()
        return loss.clamp_min(1.0E-7)
    