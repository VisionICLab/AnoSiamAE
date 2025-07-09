from typing import *
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.metrics.metrics import ADMetrics
import numpy as np
from torchvision.utils import save_image, make_grid
import os

class _BaseEngine(nn.Module):
    """ Custom Base Pytorch module for Reconstruction-based anomaly detection models
    """
    def __init__(self, cfg: dict, device:str="cuda"):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.metrics = cfg["EVALUATION"]["METRICS"]
        self.ad_metric = ADMetrics()

    def anomaly_score(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """ Anomaly scoring method

        Args:
            x (torch.Tensor): Input image [B,C,H,W]
            mask (torch.Tensor, optional): Mask of the region of interest [B,1,H,W]
        Returns:
            torch.Tensor: Anomaly score for the given image [B,]
        """
        raise NotImplementedError("Error. anomaly_score method must be implemented in subclass.")

    def reconstruct(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """ Image reconstruction method

        Args:
            x (torch.Tensor): Input image [B,C,H,W]
            mask (torch.Tensor, optional): Mask of the region of interest [B,1,H,W]
        Returns:
            torch.Tensor: Reconstructed input tensor [B,C,H,W]
        """
        raise NotImplementedError("Error. reconstruct method must be implemented in subclass.")
    
    def train_model(self, epoch:int, train_data:DataLoader, output_dir:str, visualization:bool=False, save_frq:int=1):
        raise NotImplementedError("Error. train_model method must be implemented in subclass.")

    def eval_model(self,
                   epoch:int,
                   val_data:DataLoader=None,
                   Test:bool=False,
                   output_dir:str=None) -> dict:
        """ Anomaly detection evaluation loop

        Args:
            epoch (int): Current epoch
            val_data (DataLoader, optional): Validation dataloader on which to perform the evaluation.
            Test (bool, optional): _description_. Is test data for plotting.
            output_dir (str, optional): _description_. Output directory

        Returns:
            dict: evaluation dictionnary
        """
        
        self.eval()        
        with torch.no_grad():
            labels = []
            scores = []
            for i,X in enumerate(tqdm(val_data, desc=f"Evaluation  [Epoch {epoch}]")):
                img ,label, mask  = X["image"].to(self.device), X["target"], X["mask"].to(self.device)
                rec = self.reconstruct(img, mask)
                scores.extend(self.anomaly_score(img, mask).detach().cpu())
                labels.extend(label.tolist())
                # Visualization
                if i==0:
                    save_size = min(5,img.shape[0])
                    save_image(make_grid(torch.cat((img[:save_size].detach().cpu(), rec[:save_size].detach().cpu()), 0), nrow=save_size), os.path.join(output_dir, f"rec_{epoch:04}.png"))
            eval_dict = self.ad_metric("anomaly_score", np.array(scores),np.array(labels),Test, output=output_dir )
        eval_dict["epoch"] = epoch
        return eval_dict