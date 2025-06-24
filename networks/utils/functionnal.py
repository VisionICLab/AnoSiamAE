import torch
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
from PIL import Image
import os
import torch
import numpy as np
import random
import yaml


### UTILS
class DictAsMember(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value
        except KeyError:
            raise AttributeError(name)
    
def set_seed(seed: int = 42) -> None:
    # Code Taken from w&b tutorial 
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_config(config_path):
    if os.path.exists(config_path):
        print("Loading configuration file found at {}".format(config_path))
        with open(os.path.join(config_path), "r+") as file:
            lines = file.readlines()
            if lines[0]=="!!python/object/new:networks.utils.functionnal.DictAsMember\n":
                file.seek(0)
                file.truncate()
                # start writing lines
                for number, line in enumerate(lines):
                    if number not in [0]:
                        file.write(line)

            file.seek(0)
            config = yaml.safe_load(file)
            nconfig=config
            if "dictitems" in config.keys():
                nconfig = config["dictitems"]
            if "state" in config.keys():
                for k in config["state"].keys():
                    nconfig[k] = config["state"][k]
        nconfig = DictAsMember(nconfig)
        return nconfig
    else:
        raise Exception("No configuration file was found at {}. Make sure the checkpoint exists".format(config_path))




### LOADING utils
def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        print("Loading model checkpoint found at {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, weights_only=False)
        if "best_checkpoint" in ckpt_path:
            return checkpoint["model"]
        return checkpoint
    else:
        raise Exception("No checkpoint was found at {}. Make sure the checkpoint exists".format(ckpt_path))

def load_model(ckpt, model, optimizer, scheduler):
    if model!=None:
        model.load_state_dict(ckpt["model"], strict=False)
    if optimizer!=None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler!=None:
        scheduler.load_state_dict(ckpt["scheduler"])



### OPTIMIZER utils

def init_optimizer(params, model):
    name = params.NAME
    model_params = model.parameters()
    if name=="SGD":
        optimizer = torch.optim.SGD(
            model_params,
            lr = params.LR,
            weight_decay = params.WEIGHT_DECAY,
            momentum = params.MOMENTUM
        )
    elif name=="Adam":
        optimizer = torch.optim.Adam(
            model_params,
            lr = params.LR,
            weight_decay = params.WEIGHT_DECAY,
            betas = params.BETAS
        )
    elif name=="AdamW":
        optimizer = torch.optim.AdamW(
            model_params,
            lr = params.LR,
            weight_decay = params.WEIGHT_DECAY
        )
    return optimizer

def init_scheduler(cfg, optimizer):
    if cfg.SCHEDULER.NAME=="Linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = cfg.SCHEDULER.START_FACTOR, end_factor = cfg.SCHEDULER.END_FACTOR, total_iters=cfg.SCHEDULER.END)
    elif cfg.SCHEDULER.NAME=="Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= cfg.SCHEDULER.END, eta_min=cfg.SCHEDULER.ETA_MIN)
    return scheduler

def segmentation_map(img, img_path, dataset = "lightx"):
    if not isinstance(img_path, list) and not isinstance(img_path, tuple):
        img_path = [img_path]
    tf = transforms.Compose([transforms.Resize(256,3), transforms.ToTensor()])
    masks = []
    labels = []
    if dataset=="MVTec":
        data_path = os.path.dirname(img_path[0]).replace("train","ground_truth").replace("test","ground_truth").replace("val","ground_truth")
        masks = []
        labels = []
        for file in img_path:
            img_name = os.path.basename(file).split(".")[0]
            gt_name = img_name + "_mask.png"
            gt_path = os.path.join(data_path,gt_name)

            if os.path.exists(gt_path):
                seg = tf(Image.open(gt_path)).unsqueeze(0).repeat(1,3,1,1)
                masks.append(seg)
                labels.append(1)
            else:
                masks.append(torch.zeros_like(img[0].detach().cpu()).unsqueeze(0))
                labels.append(0)
    else:
        db_path = os.path.dirname(img_path[0])
        seg_path = os.path.join(db_path, "segmentation_mask")
        masks = []
        labels = []
        for file in img_path:
            img_name = os.path.basename(file).split(".")[0]
            seg_masks = glob(os.path.join(seg_path, img_name+"*.*"))
            mask = torch.zeros_like(img[0].detach().cpu())
            if len(seg_masks)>0:
                for f in seg_masks:
                    seg = Image.open(f)
                    labels.extend(np.unique(seg))
                    mask+= tf(seg)
            masks.append((mask>0).float().unsqueeze(0))
    mask = torch.cat(masks, dim=0)
    return mask, np.unique(labels)
               

def set_requires_grad(net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
