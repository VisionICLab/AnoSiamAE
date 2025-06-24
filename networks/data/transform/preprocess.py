import numpy as np
import torch
from torchvision.transforms import v2 as tf
import cv2 as cv
import torch.nn as nn 
import cv2 as cv
from typing import *
import torch.nn.functional as F

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size:Union[int,tuple]=3, 
                 stride:Union[int,tuple]=1, 
                 padding:Union[int,tuple]=0):
        super().__init__()
        self.k = self._pair(kernel_size)
        self.str = self._pair(stride)
        self.pad = self._quad(padding)

    def _pair(self, k:Union[int,tuple]) -> tuple:
        if isinstance(k,int):
            return (k,k)
        elif isinstance(k, tuple):
            assert len(k)==2, ValueError(f"Error. Kernel size is {k} but should be int or 2-tuple.")
            return k
        else:
            raise ValueError(f"Error. Kernel size is {k} but should be int or 2-tuple.")
            return None
    
    def _quad(self,pad:Union[int,tuple]) -> tuple:
        if isinstance(pad, int):
            return (pad,pad,pad,pad)
        elif isinstance(pad, tuple):
            assert len(pad)==4, ValueError(f"Error. Padding is {pad} but should be int or 4-tuple.")
            return pad
        else:
            raise ValueError(f"Error. Kernel size is {pad} but should be int or 4-tuple.")
            return None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.pad, "constant", 0)
        x = x.unfold(-2,self.k[0],self.str[0]).unfold(-2, self.k[1], self.str[1])
        x = x.flatten(-2).median(-1)[0]
        return x
    
class Preprocessor(nn.Module):
    def __init__(self, cfg_data:dict):
        super().__init__()
        self.median = MedianPool2d(3,1,1)
        self.img_size = cfg_data["IMG_SIZE"]

    def scale(self, x:torch.Tensor) -> torch.Tensor:
         return tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])(x)
    
    def resize(self, x:torch.Tensor) -> torch.Tensor:
         return tf.Resize((self.img_size,self.img_size), interpolation=0, antialias=True)(x)

    def remove_borders(self, image: torch.Tensor) -> torch.Tensor:
        # Grayscale
        grey = image[0].unsqueeze(0)
        grey = tf.Resize(self.img_size, interpolation=0, antialias=True)(grey)

        # Binarize
        mask = self.median((grey>0.06).int())
        contours, _ = cv.findContours(mask.permute(1,2,0).numpy().astype(np.uint8)*255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        max_area=0
        best_ct = contours[0]
        for ct in contours:
            if cv.contourArea(ct)>max_area:
                best_ct = ct
        mask = torch.tensor(cv.drawContours(np.zeros_like(mask.permute(1,2,0).numpy()), [best_ct], -1, (1.0,1.0,1.0), -1 )).permute(2,0,1)
        mask = tf.Resize(min(image.shape[1:]), interpolation=0, antialias=False)(mask).float()

        lines = torch.argmax(mask, dim=1)[0]
        columns = torch.argmax(mask, dim=2)[0]
        x0,x1 = lines.nonzero()[0], lines.nonzero()[-1]
        y0,y1 = columns.nonzero()[0], columns.nonzero()[-1]
        
        # Crop image
        mask = mask[:,max(0,int(y0)):min(int(y1), image.shape[1]), max(0,int(x0)):min(int(x1), image.shape[2])]
        image = image[:,
                    max(0,int(y0)):min(int(y1), image.shape[1]),
                    max(0,int(x0)):min(int(x1), image.shape[2])]*mask
        return {"image": image, "mask": mask}
    
    def __call__(self, image:torch.Tensor) -> dict:
        image = self.scale(image)
        preprocess = self.remove_borders(image)
        return {"image": self.resize(preprocess["image"]), "mask":self.resize(preprocess["mask"])}