import torch
import torch.nn as nn
from src.data.transform.augment import AugmentData
from .masking import RandomMaskGenerator
from .blending import RandomBlender
from .anomaly_retriever import RandomAnomalyRetriever
import random
from pietorch import blend as pblend

class PatchTransform(nn.Module):
    """ Custom Pytorch Module for Pseudo-Anomaly Generation
    """
    def __init__(self, cfg_pag:dict, device:str):
        super().__init__()
        self.max_patch = cfg_pag["MASKING"]["MAX_PATCH"]
        self.ratio = cfg_pag["MASKING"]["RATIO"]
        self.blending_method = cfg_pag["BLENDING"]["METHOD"]
        self.masking_method = cfg_pag["MASKING"]["METHOD"]

        self.masking = RandomMaskGenerator(cfg_pag["MASKING"])
        self.ano_retriever = RandomAnomalyRetriever(cfg_pag["ANOMALY"])
        self.blending = RandomBlender(cfg_pag["BLENDING"])
        self.augment = AugmentData(cfg_pag["AUGMENT"],is_training=True)
        self.device = device

    def _get_random_coords(self, batch_size:int, height:int, width:int, n_coords:int, ratio:tuple[float,float]) -> torch.Tensor:
        ratio_x = (torch.rand(batch_size,n_coords)*(ratio[1]-ratio[0])+ratio[0]).to(self.device)
        ratio_y = (torch.rand(batch_size,n_coords)*(ratio[1]-ratio[0])+ratio[0]).to(self.device)

        x0 = (torch.rand(batch_size, n_coords).to(self.device)*(width*(1-ratio_x))).round()
        y0 = (torch.rand(batch_size,n_coords).to(self.device)*(height*(1-ratio_y))).round()

        return x0, y0, ratio_x, ratio_y
    
    def rectangle_process(self, target:torch.Tensor) ->dict:
        b,c,h,w = target.shape

        # Generate random patch coordinates
        n_patch = random.randint(1,self.max_patch)
        x0, y0, ratio_x, ratio_y = self._get_random_coords(b,h,w,n_patch,self.ratio) # [batch_size, n_patch]

        # Retrieve anomaly source image 
        anomalies = self.ano_retriever(target) #[batch, channel, height, width]

        return self.patch_transform(target, anomalies, x0, y0, ratio_x, ratio_y)
    
    def perlin_poisson_process(self,target:torch.Tensor) ->dict:
        if len(target.shape)==3:
            target = target.unsqueeze(0)
        b,c,h,w = target.shape
        # Generate pseudo-anomaly mask region from target tensor
        mask = self.masking(target) #[n_patch, batch, channel, height, width]
        n_patch = mask.shape[0]
        # Retrieve anomaly source image 
        anomalies = self.ano_retriever(target) #[batch, channel, height, width]
        anomalies = anomalies.unsqueeze(0).repeat(n_patch,1,1,1,1).reshape(b*n_patch,c,h,w) #[batch*n_patch, channel, height, wwidth]
        # Random patch augment
        anomalies = torch.stack([torch.clamp(self.augment(x),0,1.0) for x in anomalies]).reshape(n_patch, b, c, h, w) #[n_patch, batch, channel, height, width]
        # Blending
        pseudo_anomaly = self.blending(target, anomalies, mask)

        return torch.clamp(pseudo_anomaly,0.0,1.0), (mask.sum(0)>0).float().mean(1,keepdim=True)

        
    def patch_transform(self, target:torch.Tensor, source:torch.Tensor, x0:torch.Tensor, y0:torch.Tensor, ratio_x:torch.Tensor, ratio_y:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros_like(target)
        output = target.clone()
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                x00, x01 = x0[i,j].int(), (x0[i,j] + ratio_x[i,j]*source.shape[2]).int()
                y00, y01 = y0[i,j].int(), (y0[i,j] + ratio_y[i,j]*source.shape[2]).int()

                patch = source[i,:, x00:x01, y00:y01]
                patch = torch.clamp(self.augment(patch),0.0,1.0)
                if self.blending_method=="none":
                    output[i,:, x00:x01, y00:y01] = patch
                elif self.blending_method == "alpha":
                    alpha = 0.1+torch.rand(1).to(output.device)*0.9
                    target_patch = output[i,:, x00:x01, y00:y01]
                    output[i,:, x00:x01, y00:y01] = alpha*patch + (1-alpha)*target_patch
                elif self.blending_method == "poisson":
                    output[i] = pblend(output[i].cpu(), patch.cpu(), torch.ones_like(patch)[0].cpu(), torch.tensor([x00,y00]).cpu(),True, channels_dim=0)
                mask[i,:, x00:x01, y00:y01] = torch.ones_like(patch)
        return output, mask[:,0].unsqueeze(1)

    def forward(self, target:torch.Tensor) -> dict:
        """ Pseudo-anomaly Generation

        Args:
            target (torch.Tensor): Target image input [batch, channel, height ,width]

        Returns:
            dict:
                - "pseudo-anomaly": generated pseudo-abnormal tensor from target tensor
                - "mask": Mask of pseudo anomalies regions
        """
        if self.blending_method=="poisson" or self.masking_method=="perlin":
            return self.perlin_poisson_process(target)
        else:
            return self.rectangle_process(target)