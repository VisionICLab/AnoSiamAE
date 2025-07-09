import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import src.utils.functionnal as utils
import src.loss.manager as loss
from src.loss.regularizer import KLLoss
from src.loss.reconstruction import MSELoss
from src.models.build import build_arch
from .base import _BaseEngine
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register("VAEGAN")
class VAEGANTrainer(_BaseEngine):
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
        self.kl_criterion = KLLoss(device, "mean")
        self.kl_weight = cfg.LOSS_KL.WEIGHT
        self.adv_weight = cfg.LOSS_ADV.WEIGHT
        self.layer = cfg.LOSS_ADV.LAYER
        self.MSE = MSELoss(device=device, reduction="mean")

        # Anomaly score
        self.as_func = loss.GeneratorLossManager(cfg.ANOMALY_SCORE, device=device)


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
        loss_dict = {"generator_loss":0, "critic_loss":0, "adversarial_loss":0, "kl_loss":0}
        iter_count = {"generator":0, "critic":0}
        self.generator.train()
        self.critic.train()
        pbar = tqdm(train_data, desc="Training [Epoch {}]".format(epoch))
        for i,X in enumerate(pbar):
            x = X["image"].to(self.device)
            y, mu, logvar = self.generator(x)

            # Generator loss
            utils.set_requires_grad(self.critic, False)

            self.optimizerG.zero_grad()
            l_rec = self.criterionG(x, y)
            loss = 0
            for k in l_rec.keys():
                if k+"_loss" not in loss_dict.keys():
                    loss_dict[k + "_loss"] = l_rec[k].item()
                else:
                    loss_dict[k + "_loss"]+= l_rec[k].item()
                loss += l_rec[k]

            gen_feats = self.critic.get_intermediate_layers(y)[self.layer]
            real_feats = self.critic.get_intermediate_layers(x)[self.layer]
            l_adv = self.MSE(gen_feats, real_feats)*self.adv_weight
            loss_dict["adversarial_loss"] += l_adv.item()
            loss += l_adv

            l_kl = self.kl_criterion(mu, logvar)*self.kl_weight
            loss_dict["kl_loss"] += l_kl.item()
            loss+=l_kl

            loss.backward()
            self.optimizerG.step()
            loss_dict["generator_loss"] += loss.item()

            iter_count["generator"] += 1
            for _ in range(self.n_critic):
                # critic loss
                utils.set_requires_grad(self.critic, True)

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
            pbar.set_postfix({"critic_loss":loss_dict["critic_loss"]/iter_count["critic"], "generator_loss":loss_dict["generator_loss"]/iter_count["generator"]})
        train_dict={"epoch": epoch}
        for k in loss_dict.keys():
            if "critic" in k:
                train_dict[k] = loss_dict[k]/iter_count["critic"]
            else:
                train_dict[k] = loss_dict[k]/iter_count["generator"]
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
                }
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
        rec = self.reconstruct(x, mask)
        loss = self.as_func(x, rec)
        for i,v in enumerate(loss.values()):
            if i==0:
                res =v
            else:
                res+=v 
        return v
    

    