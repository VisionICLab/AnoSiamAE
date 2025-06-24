import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import networks.utils.functionnal as utils
import networks.loss.manager as loss
import networks.models.builder as builder
from .base import _BaseModel
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register("DAE")
class DAETrainer(_BaseModel):
    def __init__(self, cfg, device="cuda", ckpt_path=None):
        super().__init__(cfg, device)
        # Generator
        self.generator = builder.__dict__["build_"+cfg.GENERATOR.NAME](cfg.GENERATOR, device)
        
        # Loss
        self.criterionG = loss.GeneratorLossManager(cfg.LOSS_REC, device=device)
        self.noise_factor = cfg["GENERATOR"]["NOISE_FACTOR"]

        # Anomaly score
        self.as_func = loss.GeneratorLossManager(cfg.ANOMALY_SCORE, device=device)

        # Optimizer
        self.optimizerG = utils.init_optimizer(cfg.OPTIMIZER_G, self.generator)

        # Scheduler
        if cfg.SCHEDULER.NAME!=None:
            self.schedulerG = utils.init_scheduler(cfg, self.optimizerG)
        else:
            self.schedulerG=None
        self.start_epoch = 0
        if ckpt_path!=None:
            ckpt = utils.load_checkpoint(ckpt_path)
            self.start_epoch = ckpt["epoch"]
            utils.load_model(ckpt["generator"], self.generator, self.optimizerG, self.schedulerG)

    def train_model(self, epoch, train_data, output_dir, visualization=False, save_frq=1):
        loss_dict = {"generator_loss":0}
        self.generator.train()
        pbar = tqdm(train_data, desc="Training [Epoch {}]".format(epoch))
        for i,X in enumerate(pbar):
            x = X["image"].to(self.device)
            noise = torch.randn_like(x)

            y = self.generator(torch.clamp(x+self.noise_factor*noise,0.0,1.0))
            # Generator loss

            self.optimizerG.zero_grad()
            l_rec = self.criterionG(x, y)
            errREC = 0
            for k in l_rec.keys():
                if k+"_loss" not in loss_dict.keys():
                    loss_dict[k + "_loss"] = l_rec[k].item()
                else:
                    loss_dict[k + "_loss"]+= l_rec[k].item()
                errREC += l_rec[k]
            errREC.backward()
            self.optimizerG.step()
            loss_dict["generator_loss"] += errREC.item()
            
            
            if (epoch==0 or epoch%save_frq==0) and visualization and i==len(train_data)-1:
                save_size = min(x.shape[0], 5)
                save_image(make_grid(torch.cat((x[:save_size].detach().cpu(), y[:save_size].detach().cpu()), 0), nrow=save_size), os.path.join(output_dir, f"rec_{epoch:04}.png"))
            pbar.set_postfix({"generator_loss":loss_dict["generator_loss"]/(i+1)})
        train_dict={"epoch": epoch}
        for k in loss_dict.keys():
            train_dict[k] = loss_dict[k]/len(train_data)

        save_dict = {
            'cfg': self.cfg,
            "generator":{
                'model': self.generator.state_dict(),
                'optimizer': self.optimizerG.state_dict(),
                },
            'epoch': epoch + 1
            }
        if self.schedulerG!=None:
            self.schedulerG.step()
            save_dict["generator"]["scheduler"] = self.schedulerG.state_dict()
        return train_dict, save_dict
    
    def reconstruct(self, x, mask=None):
        noise = torch.randn_like(x)
        rec = self.generator(torch.clamp(x+self.noise_factor*noise,0.0,1.0))
        if not mask is None:
            rec = rec*mask
        return rec
    
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        rec = self.reconstruct(x, mask)
        loss = self.as_func(x, rec)
        for i,v in enumerate(loss.values()):
            if i==0:
                res =v
            else:
                res+=v 
        return v

    