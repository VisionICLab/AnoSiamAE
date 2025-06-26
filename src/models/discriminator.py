import torch.nn as nn
from .build import ARCH_REGISTRY

@ARCH_REGISTRY.register("discriminator")
class Discriminator(nn.Module):
    def __init__(self, cfg_model:dict):
        super(Discriminator, self).__init__()
        in_shape = cfg_model["IN_SHAPE"]
        num_channels = cfg_model["NUM_CHANNELS"]
        dropout = cfg_model["DROPOUT"]
        in_shape = cfg_model["IN_SHAPE"]
        norm = cfg_model["NORM"]
        num_groups = cfg_model["NUM_GROUPS"]

        self.patch_disc = cfg_model["PATCH"]
        bias = True
        if norm=="batch":
            bias=False

        self.layers = nn.Sequential()
        out_channel, w, h = in_shape
        for i,c in enumerate(num_channels):
            in_channel = out_channel
            out_channel = c
            if i<len(num_channels)-1:
                kernel = 4
                stride=2
                w, h = w//2, h//2 
            else:
                kernel = 4
                stride=1
                w, h = w-1, h-1
            if i>0:
                self.layers.add_module("conv{}".format(i+1),nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, padding=1, bias=bias))
                if norm=="batch":
                    self.layers.add_module("norm{}".format(i+1), nn.BatchNorm2d(out_channel))
                elif norm=="group":
                    self.layers.add_module("norm{}".format(i+1), nn.GroupNorm(num_groups=num_groups, num_channels=out_channel))
                elif norm=="layer":
                    self.layers.add_module("norm{}".format(i+1), nn.GroupNorm(num_groups=out_channel, num_channels=out_channel))
            else:
                self.layers.add_module("conv{}".format(i+1),nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, padding=1))
            self.layers.add_module("act{}".format(i+1),nn.SiLU())
            self.layers.add_module("drop{}".format(i+1),nn.Dropout(dropout))
        
        if self.patch_disc:
            self.layers.add_module("conv{}".format(i+2),nn.Conv2d(in_channels=out_channel, out_channels=1, kernel_size=4, stride=1, padding=1))
        else:
            self.layers.add_module("conv{}".format(i+2),nn.Conv2d(in_channels=out_channel, out_channels=1, kernel_size=5, stride=1, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wasserstein:bool=False):
        for layer in self.layers:
            x = layer(x)
        if wasserstein:
            return x.view(x.shape[0],-1)
        return self.sigmoid(x.view(x.shape[0],-1))

    
    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad
        
    def get_intermediate_layers(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.SiLU):
                features.append(x)
        return features

    def predict(self,x):
        return self.sigmoid(self(x))

