import torch
import torch.nn as nn
import numpy as np
from pietorch import blend as pblend


def poisson_blending(target:torch.Tensor, aug_source:torch.Tensor, mask:torch.Tensor, mix_gradients:bool) ->torch.Tensor:
    """ Perform poisson of source image into target image depending on given mask
    Based on https://github.com/matt-baugh/pytorch-poisson-image-editing

    Args:
        target (torch.Tensor): target tensor [channel, height, width]
        aug_source (torch.Tensor): augmented anomaly tensor [channel, height, width]
        mask (torch.Tensor): masked tensor [channel, height, width]
    Returns:
        torch.Tensor: blended tensor
    """
    target, aug_source, mask = target.detach().cpu(), aug_source.detach().cpu(), mask.detach().cpu()
    target = pblend(target, aug_source, mask[0], torch.tensor([0,0]), mix_gradients, channels_dim=0)
    return target

def alpha_blending(target:torch.Tensor, aug_source:torch.Tensor, mask:torch.Tensor, alpha:torch.Tensor) ->torch.Tensor:
    """ Simple alpha factor blending computation

    Args:
        target (torch.Tensor): target tensor [batch, channel, height, width]
        aug_source (torch.Tensor): augmented anomaly tensor [n_patch, batch, channel, height, width]
        mask (torch.Tensor): masked tensor [n_patch, batch, channel, height, width]
        alpha (float): blending factor [n_patch, batch, 1, 1, 1]
    Returns:
        torch.Tensor: blended tensor
    """
    source = (((1-alpha)*target + alpha*aug_source)*mask).sum(0)
    mask = mask.sum(0)
    source[mask!=0] = source[mask!=0]/mask[mask!=0]
    mask = (mask>0).int()    
    return (1-mask)*target + source

class RandomBlender(nn.Module):
    """ Custom Pytorch Module for Random Tensor Blending
    """
    def __init__(self, params:dict):
        super().__init__()
        self.params = params
        self.method = params["METHOD"]

    def forward(self, target:torch.Tensor, aug_source:torch.Tensor, mask:torch.Tensor) ->torch.Tensor:
        """ Blending based on specified parameters

        Args:
            target (torch.Tensor): target tensor [batch, channel, height, width]
            aug_source (torch.Tensor): augmented anomaly tensor [n_patch, batch, channel, height, width]
            mask (torch.Tensor): masked tensor [n_patch, batch, channel, height, width]

        Returns:
            torch.Tensor: blended tensor
        """
        if self.method == "none":
            source = torch.zeros(aug_source.shape[1:]).to(target.device)
            for i in range(mask.shape[0]):
                source[mask[i]!=0] = (aug_source[i]*mask[i])[mask[i]!=0]
            mask = (mask.sum(0)>0).int()
            return (1-mask)*target + source
        elif self.method == "alpha":
            alpha = 0.1+torch.rand((mask.shape[0],mask.shape[1],1,1,1)).to(target.device)*0.9
            return alpha_blending(target, aug_source, mask,alpha)
        elif self.method == "poisson":
            return torch.cat([poisson_blending(target[i], aug_source[0,i], (mask.sum(0)>0).float()[i], True).unsqueeze(0) for i in range(target.shape[0])], dim=0).to(target.device)
        else:
            raise KeyError(f"Error. {self.method} is not supported.")

