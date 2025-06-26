import argparse

import numpy as np
import src.utils.functionnal as utils
from src.config.config import load_cfg
from src.engine import build_model
from trainer import Trainer
import os 


if __name__=="__main__":
    parser = argparse.ArgumentParser('Training parameters')
    parser.add_argument('--folder', default=None, type=str, help='Path to model foler')
    parser.add_argument('--cfg_path', default="config/cfg.yaml", type=str, help='Path to configuration file')
    parser.add_argument('--test_only',action='store_true', default=False, help='Skip training if True')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to pretrained model')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--output', default=None, type=str, help='Overide cfg output saving directory')
    parser.add_argument('--data', default="rfmid", type=str, help='Dataset to be used')
    parser.add_argument('--batch_size', default=8, type=int, help='Dataset to be used')

    args = parser.parse_args()
    if not args.folder is None:
        cfg_path = os.path.join(args.folder, "cfg.yaml")
        ckpt_path = os.path.join(args.folder, "model", "checkpoint.pth")
    else:
        cfg_path = args.cfg_path
        ckpt_path = args.ckpt_path
    cfg = load_cfg(cfg_path)
    utils.set_seed(cfg["SYSTEM"]["SEED"])

    if args.output!=None:
        print("Overriding output saving directory at ", args.output)
        cfg.OUTPUT = args.output

    cfg["DATA"]["NAME"] = args.data
    cfg["DATA"]["BATCH_SIZE"] = args.batch_size

    model = build_model(cfg, device=args.device, ckpt_path=ckpt_path)
    model_parameters = filter(lambda p: p.requires_grad, model.generator.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters: ", params)

    t = Trainer(model, cfg, test_only=args.test_only)
    t.run()
