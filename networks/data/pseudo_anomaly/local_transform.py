from torchvision.transforms import v2 as transforms
import torch
import torch.nn as nn
import random
class Null():
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return torch.zeros_like(x)

class Transform(object):
    def __init__(self, pjitter:float=0.3,
                 pgauss:float=0.5,
                 pdrop:float=0.3,
                 pel:float=0.1,
                 phflip:float=0.2,
                 pvflip:float=0.2,
                 pnull:float=0.1,
                 ) -> None:
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=None, contrast=1.0, saturation=1.0, hue=0.5)],
                p=pjitter
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur((5,9), random.uniform(10, 100))],
                p=pgauss
            ),
            transforms.RandomApply(
                [nn.Dropout(0.02)],
                p=pdrop
            ),
            transforms.RandomApply(
                [transforms.ElasticTransform(250, sigma=5)],
                p=pel
            ),
            transforms.RandomHorizontalFlip(p=phflip),
            transforms.RandomVerticalFlip(p=pvflip),
            transforms.RandomApply(
                [Null()],
                p=pnull
            ),
        ])
        
        pass

    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class LocalTransform(object):
    def __init__(self, params, device:str="cuda"):
        self.device = device
        self.ratio = params["RATIO"]
        self.max_patch = params["N_PATCH"]
        self.interval = params["STEP"]
        self.transform = Transform(pjitter=params["JITTER"],
                                   pgauss=params["JITTER"],
                                   pdrop=params["DROP"],
                                   pel=params["ELASTIC"],
                                   phflip=params["HFLIP"],
                                   pvflip=params["VFLIP"],
                                   pnull=params["ZERO"])
        pass

    def augment(self, img:torch.Tensor, loc_h:torch.Tensor, loc_w:torch.Tensor, nh:torch.Tensor, nw:torch.Tensor) -> torch.Tensor:
        aug_img = img.clone()
        mask = torch.zeros_like(img)
        for b in range(img.shape[0]):
            for p in range(loc_h.shape[1]):
                if loc_h[b,p]!=-1 or loc_w[b,p]!=-1:
                    beta = random.uniform(0.1, 1.0)
                    patch = img[b,:,loc_h[b,p]:loc_h[b,p]+nh[b,p],loc_w[b,p]:loc_w[b,p]+nw[b,p]]
                    patch = self.transform(patch)
                    aug_img[b,:,loc_h[b,p]:loc_h[b,p]+nh[b,p],loc_w[b,p]:loc_w[b,p]+nw[b,p]] = beta*patch + (1-beta)*aug_img[b,:,loc_h[b,p]:loc_h[b,p]+nh[b,p],loc_w[b,p]:loc_w[b,p]+nw[b,p]]
                    mask[b,:,loc_h[b,p]:loc_h[b,p]+nh[b,p],loc_w[b,p]:loc_w[b,p]+nw[b,p]] = torch.ones_like(patch)
        return aug_img, mask
    
    def __call__(self, img: torch.Tensor,  epoch:int, mask:torch.Tensor=None, enforce_augment:str=False) -> torch.Tensor:
        """_summary_

        Args:
            img (torch.Tensor): Tensor image in shape [B,C,H,W]

        Returns:
            torch.Tensor: Transformed image in shape [B,C,H,W]
        """
        if self.interval ==0:
            n_patch=self.max_patch
        else:
            n_patch = min(self.max_patch, epoch//self.interval+1)

        b,c,h,w = img.shape

        # Define random area ratios to crop
        ratio_h = torch.rand(b,n_patch)*(self.ratio[1]-self.ratio[0]) + self.ratio[0]
        ratio_w = torch.rand(b,n_patch)*(self.ratio[1]-self.ratio[0]) + self.ratio[0]

        nh = (ratio_h*h).round().int()
        nw = (ratio_w*w).round().int()

        # Define random locations to crop
        loc_h = (torch.rand(b, n_patch)*(h-nh)).round().int()
        loc_w = (torch.rand(b, n_patch)*(w-nw)).round().int()

        # Randomly select between 0 and max_patch
        if enforce_augment:
            mask_patch = (torch.rand(b,n_patch-1)>0.5).int()
            mask_patch = torch.cat((torch.ones((b,1)), mask_patch), 1)
        else:
            mask_patch = (torch.rand(b,n_patch)>0.5).int()
        loc_h = torch.where(mask_patch==1, loc_h, -1)
        loc_w = torch.where(mask_patch==1, loc_w, -1)

        aug_img, aug_mask = self.augment(img, loc_h, loc_w, nh, nw)
        if not mask is None:
            return aug_img*mask, aug_mask*mask
        return torch.clamp(aug_img,0.0,1.0), aug_mask