import torch.nn as nn
from torchvision.transforms import v2 as tf
import random
from PIL import Image
import torch
from typing import *
import torch


class Eraser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
class AugmentData(nn.Module):
    """ Custom pytorch data augmentation module
    """
    def __init__(self, cfg_augment:dict, is_training:bool):
        super().__init__()
        self.cfg = cfg_augment
        self.transform = self.load_transform()
        self.is_training = is_training
        
    def load_transform(self):
        """Load and compose augmentations from cfg file 

        Args:
            is_training (bool): Apply destructive augmentations only on training samples
        """
        transform = []
        transform.append(tf.RandomApply([tf.RandomRotation(degrees=(0,360), interpolation=Image.NEAREST)], p=self.cfg["ROT"]))
        transform.append(tf.RandomHorizontalFlip(self.cfg["HFLIP"]))
        transform.append(tf.RandomVerticalFlip(self.cfg["VFLIP"]))
        transform.append(tf.RandomApply([tf.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)], p=self.cfg["CJITT"]))
        transform.append(tf.RandomApply([tf.GaussianBlur(5, random.uniform(0.1, 1))], p=self.cfg["GAUSS"]))            
        transform.append(tf.RandomApply([nn.Dropout(p=0.02)], p=self.cfg["DROP"]))
        transform.append(tf.RandomApply([Eraser()], p=self.cfg["ERASE"]))
        transform.append(tf.RandomPosterize(bits=2, p=self.cfg["POSTERIZE"]))
        transform.append(tf.RandomAdjustSharpness(sharpness_factor=2, p=self.cfg["SHARPNESS"]))
        transform.append(tf.RandomAutocontrast(p=self.cfg["CONTRAST"]))
        transform.append(tf.RandomEqualize(p=self.cfg["EQUALIZE"]))
        transform.append(tf.RandomSolarize(threshold=0.8, p=self.cfg["SOLARIZE"]))
        transform.append(tf.RandomApply([tf.ElasticTransform(250, sigma=5)], p=self.cfg["ELASTIC"]))
        return tf.Compose(transform)

    def forward(self, *args:torch.Tensor) -> torch.Tensor:
        """ Apply transformations on tensor

        Args:
            *args (torch.Tensor): Input tensors [channel, height, width]
        Returns:
            torch.Tensor: Augmented tensor
        """

        if self.is_training:
            cat = self.transform(torch.cat([x.unsqueeze(0) for x in args],dim=0))
            outputs = tuple(cat[i] for i in range(len(args)))
            return outputs if len(outputs)>1 else outputs[0]
        return args if len(args)>1 else args[0]