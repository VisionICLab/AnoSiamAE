from networks.utils.registry import Registry
from .access_data import get_data_path
from .base import _BaseDataloader

DATALOADER_REGISTRY = Registry()

def build_dataloader(cfg:dict) -> _BaseDataloader:
    data_path = get_data_path(cfg["DATA"]["NAME"])
    dataloader = DATALOADER_REGISTRY[cfg["DATA"]["NAME"]](data_path, cfg)
    return dataloader