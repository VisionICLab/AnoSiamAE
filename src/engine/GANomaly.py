import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import src.utils.functionnal as utils
import src.loss.manager as loss
from src.loss.reconstruction import MSELoss, MAELoss
from src.models.build import build_arch
from .base import _BaseModel
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register("GANomaly")
class GANomalyTrainer(_BaseModel):
    def __init__(self, cfg, device="cuda", ckpt_path=None):
        super().__init__(cfg, device)
        # Critic
        self.critic = build_arch(cfg["CRITIC"], device)
        self.n_critic = self.cfg.CRITIC.N_CRITIC # N discriminator forward per generator forward

        # Generator
        self.generator = build_arch(cfg["GENERATOR"], device)
        
        # Loss
        self.criterionG = loss.GeneratorLossManager(cfg.LOSS_REC, device)
        self.criterionC = loss.AdversarialLossManager(cfg.LOSS_CRITIC, device)
        self.MSE = MSELoss(device=device, reduction="mean")
        self.MAE = MAELoss(device=device, reduction="none")
        self.adv_weight = cfg.LOSS_ADV.WEIGHT
        self.layer = cfg.LOSS_ADV.LAYER
        self.enc_weight = cfg.LOSS_ENC.WEIGHT

        # Optimizer
        self.optimizerC = utils.init_optimizer(cfg.OPTIMIZER_C, self.critic)
        self.optimizerG = utils.init_optimizer(cfg.OPTIMIZER_G, self.generator)

        # Scheduler
        if cfg.SCHEDULER.NAME!=None:
            self.schedulerG = utils.init_scheduler(cfg, self.optimizerG)
            self.schedulerC = utils.init_scheduler(cfg, self.optimizerC)
        else:
            self.schedulerG=None
            self.schedulerC=None
        self.start_epoch = 0
        if ckpt_path!=None:
            ckpt = utils.load_checkpoint(ckpt_path)
            self.start_epoch = ckpt["epoch"]
            utils.load_model(ckpt["generator"], self.generator, self.optimizerG, self.schedulerG)
            utils.load_model(ckpt["critic"], self.critic, self.optimizerC, self.schedulerC)

    def train_model(self, epoch, train_data, output_dir, visualization=False, save_frq=1):
        loss_dict = {"generator_loss":0, "critic_loss":0, "adversarial_loss":0, "encoder_loss":0}
        iter_count = {"critic":0}
        self.generator.train()
        self.critic.train()
        pbar = tqdm(train_data, desc="Training [Epoch {}]".format(epoch))
        for i,X in enumerate(pbar):
            x = X["image"].to(self.device)
            y, z, z_enc = self.generator(x)

            # Generator loss
            self.optimizerG.zero_grad()
            loss = 0
            
            gen_feats = self.critic.get_intermediate_layers(y)[self.layer]
            real_feats = self.critic.get_intermediate_layers(x)[self.layer]
            l_adv = self.MSE(gen_feats, real_feats)*self.adv_weight
            loss_dict["adversarial_loss"] += l_adv.item()
            loss += l_adv

            l_rec = self.criterionG(x, y)
            for k in l_rec.keys():
                if k+"_loss" not in loss_dict.keys():
                    loss_dict[k + "_loss"] = l_rec[k].item()
                else:
                    loss_dict[k + "_loss"]+= l_rec[k].item()
                loss += l_rec[k]

            l_enc = self.MSE(z_enc, z)*self.enc_weight
            loss_dict["encoder_loss"] += l_enc.item()
            loss += l_enc

            loss.backward()
            self.optimizerG.step()
            loss_dict["generator_loss"] += loss.item()

            # critic loss
            for _ in range(self.n_critic):
                self.optimizerC.zero_grad() 
                outC_real = self.critic(x)
                errC_real = self.criterionC(outC_real, is_real = True, for_disc = True)
                outC_fake = self.critic(y.detach())
                errC_fake = self.criterionC(outC_fake, is_real = False, for_disc = True)
                errC = (errC_real + errC_fake)*0.5
                errC.backward()
                self.optimizerC.step()
                loss_dict["critic_loss"] += errC.item()
                iter_count["critic"] += 1


            if (epoch==0 or epoch%save_frq==0) and visualization and i==len(train_data)-2:
                save_size = min(x.shape[0], 5)  
                save_image(make_grid(torch.cat((x[:save_size].detach().cpu(), y[:save_size].detach().cpu()), 0), nrow=save_size), os.path.join(output_dir, f"rec_{epoch:04}.png"))
            pbar.set_postfix({"critic_loss":loss_dict["critic_loss"]/iter_count["critic"], "generator_loss":loss_dict["generator_loss"]/(i+1), "encoder_loss":loss_dict["encoder_loss"]/(i+1)})
        train_dict={"epoch": epoch}
        for k in loss_dict.keys():
            if "critic" in k:
                train_dict[k] = loss_dict[k]/iter_count["critic"]
            else:
                train_dict[k] = loss_dict[k]/len(train_data)
        save_dict = {
            "epoch":epoch+1,
            "cfg":self.cfg,
            "generator":{
                'model': self.generator.state_dict(),
                'optimizer': self.optimizerG.state_dict(),
                },
            "critic":{
                'model': self.critic.state_dict(),
                'optimizer': self.optimizerC.state_dict(),
                },
            }
        if self.schedulerG!=None:
            self.schedulerG.step()
            self.schedulerC.step()
            save_dict["generator"]["scheduler"] = self.schedulerG.state_dict()
            save_dict["critic"]["scheduler"] = self.schedulerC.state_dict()
        return train_dict, save_dict
    
    def reconstruct(self, x, mask=None):
        rec = self.generator.reconstruct(x)
        if not mask is None:
            rec = rec*mask
        return rec

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        _,z,z_enc = self.generator(x)
        return self.MAE(z, z_enc)
    

    