from yacs.config import CfgNode as CN
import os

_C = CN()
_C.OUTPUT = "output/default"
_C.NETWORK = "AE"

# System parameters
_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.SEED = 42

# Data parameters
_C.DATA = CN()
_C.DATA.NAME = "lightx"
_C.DATA.VAL_NORM_RATIO = 0.2
_C.DATA.BATCH_SIZE = 16
_C.DATA.EPOCHS = 150
_C.DATA.IMG_SIZE = 256

# Data augmentation hyperparameters
_C.DATA.AUGMENT = CN()
_C.DATA.AUGMENT.HFLIP = 0.0
_C.DATA.AUGMENT.VFLIP = 0.0
_C.DATA.AUGMENT.ROT = 0.0
_C.DATA.AUGMENT.CJITT = 0.0
_C.DATA.AUGMENT.GAUSS = 0.0
_C.DATA.AUGMENT.DROP = 0.0
_C.DATA.AUGMENT.ERASE = 0.0
_C.DATA.AUGMENT.POSTERIZE = 0.0
_C.DATA.AUGMENT.CONTRAST = 0.0
_C.DATA.AUGMENT.SHARPNESS = 0.0
_C.DATA.AUGMENT.EQUALIZE = 0.0
_C.DATA.AUGMENT.SOLARIZE = 0.0
_C.DATA.AUGMENT.ELASTIC = 0.0

# Data preprocessing hyperparameters
_C.PSEUDO_ANOMALY = CN()
_C.PSEUDO_ANOMALY.MASKING = CN()
_C.PSEUDO_ANOMALY.MASKING.METHOD = "rectangle"
_C.PSEUDO_ANOMALY.MASKING.MAX_PATCH = 20
_C.PSEUDO_ANOMALY.MASKING.RATIO = [0.02, 0.5]
_C.PSEUDO_ANOMALY.ANOMALY = CN()
_C.PSEUDO_ANOMALY.ANOMALY.METHOD = "same"
_C.PSEUDO_ANOMALY.BLENDING = CN()
_C.PSEUDO_ANOMALY.BLENDING.METHOD = "alpha"
_C.PSEUDO_ANOMALY.AUGMENT = CN()
_C.PSEUDO_ANOMALY.AUGMENT.HFLIP = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.VFLIP = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.ROT = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.CJITT = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.GAUSS = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.DROP = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.ERASE = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.POSTERIZE = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.CONTRAST = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.SHARPNESS = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.SOLARIZE = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.EQUALIZE = 0.0
_C.PSEUDO_ANOMALY.AUGMENT.ELASTIC = 0.0



# Evaluation parameters
_C.EVALUATION = CN()
_C.EVALUATION.TRAIN_FRQ = 2
_C.EVALUATION.VAL_FRQ = 5
_C.EVALUATION.METRICS = "all"

# Scheduler parameters
_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = "Cosine"
_C.SCHEDULER.ETA_MIN = 1.0E-4
_C.SCHEDULER.END = _C.DATA.EPOCHS


def get_cfg_defaults() -> CN:
  """Get a yacs CfgNode object with default values"""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def load_cfg(cfg_path:str) -> CN:
    """Load config from cfg_path and merge with base config

    Args:
        cfg_path (str): Path to .yaml config file

    Returns:
        CN: CfgNode
    """
    # Base config
    cfg_base = get_cfg_defaults()   

    # Merge from yaml config
    if cfg_path is not None and os.path.exists(cfg_path):
        cfg_base.set_new_allowed(True)
        cfg_base.merge_from_file(cfg_path)

    print("Config file successfully loaded.")
    return cfg_base

def save_cfg(cfg:CN, save_path:str) -> None:
    """Save config node to the specified path

    Args:
        cfg (CN): Configuration node to be saved
        save_path (str): Path to saving folder
    """
    cfg.dump(stream=open(os.path.join(save_path, "cfg.yaml"), 'w'))
    print(f"Config file successfuly saved at {save_path}.")



