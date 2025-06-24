import torch.nn as nn
from networks.data.transform.normalizer import ImageNetNorm
from torch import Tensor
from typing import *
import torchvision
import piqa
import torch
from networks.utils.constant import VGG19_WEIGHT_PATH

class LPIPSLoss(nn.Module):
    def __init__(self, device, net_type="vgg", reduction="mean"):
        super(LPIPSLoss, self).__init__()
        assert net_type in ['squeeze', 'vgg', 'alex']
        self.device = device
        self.net_type = net_type
        self.reduction = reduction
        self.criterion = piqa.LPIPS(network = self.net_type, reduction=self.reduction).to(self.device)
        
    def forward (self, img,rec):
        return self.criterion(img,rec)

    def get_name(self):
        return "LPIPSLoss"
    
class Layer(nn.Module):
    r"""Perceptual network that intercepts and returns the output of target layers
    within its foward pass.

    Args:
        layers: A list of layers.
        targets: A list of target layer indices.
    """

    def __init__(self, layers: List[nn.Module], targets: List[int]):
        super().__init__()

        self.blocks = nn.ModuleList()

        i = 0
        for j in targets:
            self.blocks.append(nn.Sequential(*layers[i : j + 1]))
            i = j + 1

    def forward(self, x: Tensor) -> List[Tensor]:
        y = []

        for block in self.blocks:
            x = block(x)
            y.append(x)

        return y

class PerceptualVGGLoss(nn.Module):
    def __init__(self, device="cuda", epsilon: float = 1e-10, reduction: str = 'mean',targets: list = [22],norm: str = "l1"):
        super().__init__()

        # ImageNet normalization
        self.normalize = ImageNetNorm()

        layers = torchvision.models.vgg19(weights='DEFAULT').features.to(device=device)
        targets = targets

        self.net = Layer(list(layers), targets)
        self.net.eval()
        # print(self.net)

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

        self.epsilon = epsilon
        self.reduction = reduction
        self.norm = norm

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # ImageNet normalization
        x = self.normalize(x)
        y = self.normalize(y)

        # LPIPS
        loss = 0
        for i,(fx, fy) in enumerate(zip(self.net(x), self.net(y))):
            if self.norm=="l1":
                diff = (fx-fy).abs()
            elif self.norm=="l2":
                diff = (fx-fy).square()
            loss += diff.mean(dim=[1,2,3])
        loss = loss/(i+1)
        if self.reduction=="mean":
            loss = loss.mean()
        elif self.reduction=="sum":
            loss = loss.mean()
        return loss
    
class RelativePerceptualVGGLoss(nn.Module):
    def __init__(self, device="cuda", epsilon: float = 1e-10, reduction: str = 'mean',targets: list = [22],norm: str = "l1"):
        super().__init__()

        self.mean_var_dict = torch.load(VGG19_WEIGHT_PATH, weights_only=False)
        # ImageNet normalization
        self.normalize = ImageNetNorm()

        layers = torchvision.models.vgg19(weights='DEFAULT').features.to(device=device)
        self.targets = targets
        self.name_layer = ["c11","r11","c12","r12","p1",
                "c21","r21","c22","r22","p2",
                "c31","r31","c32","r32","c33","r33","c34","r34","p3",
                "c41","r41","c42","r42","c43","r43","c44","r44","p4",
                "c51","r51","c52","r52","c53","r53","c54","r54","p5"]


        self.net = Layer(list(layers), targets)
        self.net.eval()

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

        self.epsilon = epsilon
        self.reduction = reduction
        self.norm = norm

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # ImageNet normalization
        x = self.normalize(x)
        y = self.normalize(y)

        # LPIPS
        loss = 0
        for i,(fx, fy) in enumerate(zip(self.net(x), self.net(y))):
            mean, var = self.mean_var_dict[self.name_layer[self.targets[i]]]
            mean, var = mean.to(fx.device), var.to(fy.device)

            norm_fx = (fx-mean)/var
            norm_fy = (fy-mean)/var
            if self.norm=="l1":
                diff = (norm_fx-norm_fy).abs()
                if self.reduction=="max":
                    div = norm_fx.abs().mean(dim=[1], keepdim=True)
                else:
                    div = norm_fx.abs().mean(dim=[1,2,3], keepdim=True)
            elif self.norm=="l2":
                diff = (norm_fx-norm_fy)**2
                if self.reduction=="max":
                    div = norm_fx.square().mean(dim=[1], keepdim=True).sqrt()
                else:
                    div = norm_fx.square().mean(dim=[1,2,3], keepdim=True).sqrt()
            if self.reduction == "max":
                loss += (diff/div).mean(dim=1)
            else:
                loss += (diff/div).mean(dim=[1,2,3])
        loss = (loss/(i+1))
        if self.reduction=="mean":
            loss = loss.mean()
        elif self.reduction=="sum":
            loss = loss.sum()
        elif self.reduction=="max":
            loss = loss.view(loss.shape[0],-1).max(dim=1).values
        return loss
