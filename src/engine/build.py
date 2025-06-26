from src.utils.registry import Registry
from src.data.access_data import get_data_path
from .base import _BaseModel

MODEL_REGISTRY = Registry()

def build_model(cfg:dict, device:str="cuda", ckpt_path:str=None) -> _BaseModel:
    model = MODEL_REGISTRY[cfg["NETWORK"]](cfg=cfg, device=device, ckpt_path=ckpt_path)
    return model