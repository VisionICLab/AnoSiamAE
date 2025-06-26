from src.utils.registry import Registry
import torch.nn as nn

ARCH_REGISTRY = Registry()

def build_arch(cfg_model:dict,device:str) -> nn.Module:
    model = ARCH_REGISTRY[cfg_model["NAME"].lower()](cfg_model)
    return model.to(device)