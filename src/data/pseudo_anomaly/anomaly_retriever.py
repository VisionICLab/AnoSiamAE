import torch
import torch.nn as nn
from torchvision.io import ImageReadMode, read_image
from glob import glob
import os
import random
from torchvision.transforms import v2 as tf

DTD_PATH = "/store/data/dtd/images"

def permute_batch(x:torch.Tensor) -> torch.Tensor:
    """ Permute batch

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: permuted tensor
    """
    tmp = x[1:]
    return torch.cat([tmp, x[0].unsqueeze(0)], dim=0)

def dtd_retriever(n_samples:int, height:int, width:int) -> torch.Tensor:
    def scale_and_resize(x:torch.Tensor) -> torch.Tensor:
        transform = []
        transform.append(tf.ToDtype(torch.float32, scale=True))
        transform.append(tf.Resize((height,width), interpolation=0, antialias=True))
        transform = tf.Compose(transform)
        return transform(x)
    texture_folder = [random.choice(glob(os.path.join(DTD_PATH, "*"))) for _ in range(n_samples)]
    files = [random.choice(glob(os.path.join(x, "*"))) for x in texture_folder]
    img = torch.stack([scale_and_resize(read_image(x, mode=ImageReadMode.RGB)) for x in files])
    img = tf.ToDtype(torch.float32, scale=True)(img)
    img = tf.Resize((height,width), interpolation=0, antialias=True)(img)
    return img

class RandomAnomalyRetriever(nn.Module):
    """ Custom Pytorch Module for Random Anomaly Retrieving
    """
    def __init__(self, params:dict):
        super().__init__()
        self.params = params
        self.method = params["METHOD"]

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.method=="same":
            return x
        elif self.method=="batch":
            return permute_batch(x)
        elif self.method=="dtd":
            return dtd_retriever(x.shape[0], x.shape[2], x.shape[3]).to(x.device)
        else:
            raise KeyError(f"Error. {self.method} is not supported.")
