import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import src.utils.functionnal as utils
from src.loss.segmentation import  FocalLoss
from src.loss.reconstruction import SSIMLoss, MSELoss

from src.data.pseudo_anomaly.generator import PatchTransform
from .base import _BaseModel
from .build import MODEL_REGISTRY
from src.models.build import build_arch

@MODEL_REGISTRY.register("DRAEM")
class DRAEMTrainer(_BaseModel):
    def __init__(self, cfg, device="cuda", ckpt_path=None):
        super().__init__(cfg, device)
        # Generator
        self.generator = build_arch(cfg["MODEL"], device)

        self.pag = PatchTransform(cfg["PSEUDO_ANOMALY"], device)

        # Loss
        self.focal = FocalLoss(alpha=0.75, reduction="mean", gamma=2)
        self.ssim = SSIMLoss(self.device, reduction="mean")
        self.mse = MSELoss(self.device, reduction="mean")

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
        loss_dict = {"generator_loss":0, "seg_loss":0, "ssim":0, "mse":0}
        self.generator.train()
        pbar = tqdm(train_data, desc="Training [Epoch {}]".format(epoch))
        for i,X in enumerate(pbar):
            x = X["image"].to(self.device)
            x_aug, target = self.pag(x)
            y_aug, seg = self.generator(x_aug)

            # Generator loss
            self.optimizerG.zero_grad()
            errG = 0

            ## Reconstruction loss            
            l_ssim = self.ssim(x, y_aug)*2
            loss_dict["ssim"] += l_ssim.item()
            errG += l_ssim

            l_mse = self.mse(x, y_aug)
            loss_dict["mse"] += l_mse.item()
            errG += l_mse

            ## Segmentation loss
            l_seg = self.focal(seg, target)
            loss_dict["seg_loss"] += l_seg.item()
            errG += l_seg
            
            errG.backward()
            self.optimizerG.step()
            loss_dict["generator_loss"] += errG.item()

            if (epoch==0 or epoch%save_frq==0) and visualization and i==len(train_data)-2:
                save_size = min(x.shape[0], 5)
                anomaly_map = torch.softmax(seg.detach(), dim=1)[:,1].unsqueeze(1)
                save_image(make_grid(torch.cat((x_aug[:save_size].detach().cpu(), y_aug[:save_size].detach().cpu(), anomaly_map[:save_size].repeat(1,3,1,1).detach().cpu(), target[:save_size].repeat(1,3,1,1).detach().cpu()), 0), nrow=save_size), os.path.join(output_dir, f"rec_aug_{epoch:04}.png"))          
            pbar.set_postfix({"generator_loss":loss_dict["generator_loss"]/(i+1), "seg_loss":loss_dict["seg_loss"]/(i+1)})
        train_dict={"epoch": epoch}
        for k in loss_dict.keys():
            train_dict[k] = loss_dict[k]/len(train_data)

        save_dict = {
            "epoch":epoch+1,
            "cfg":self.cfg,
            "generator":{
                'model': self.generator.state_dict(),
                'optimizer': self.optimizerG.state_dict(),
                },
            }
        if self.schedulerG!=None:
            self.schedulerG.step()
            save_dict["generator"]["scheduler"] = self.schedulerG.state_dict()
        return train_dict, save_dict

    def reconstruct(self, x, mask=None):
        rec = self.generator.reconstruct(x)
        if not mask is None:
            rec = rec*mask
        return rec
    
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        _, pred = self.generator(x)
        anomaly_map = torch.softmax(pred.detach(), dim=1)[:,1].unsqueeze(1)
        if not mask is None:
            anomaly_map = anomaly_map*mask
        return anomaly_map.amax(dim=(1,2,3))

    