import torch
import torch.nn as nn
from .preprocess import Preprocessor
from .augment import AugmentData

class Transform(nn.Module):
    def __init__(self, cfg_data:dict, is_training:bool):
        super().__init__()
        self.cfg_data = cfg_data
        self.augment = AugmentData(cfg_augment=cfg_data["AUGMENT"], is_training=is_training)
        self.preprocess = Preprocessor(cfg_data)

    def forward(self, image:torch.Tensor) -> dict:
        preprocess = self.preprocess(image)
        return {"image":self.augment(preprocess["image"]), "mask":preprocess["mask"]}
    
    
