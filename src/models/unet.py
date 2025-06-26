import torch.nn as nn
import torch
from .layers.blocks import customResBlock
from .build import ARCH_REGISTRY

class customDownsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class customUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsamle = nn.Upsample(scale_factor = 2, mode = 'nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.upsamle(x)
        x = self.conv(x)
        return x

@ARCH_REGISTRY.register("unet")
class UNet(nn.Module):
    def __init__(self, cfg_model:dict):
        super().__init__()
        in_channels = cfg_model["IN_CHANNEL"]
        init_features = cfg_model["INIT_FEATURES"]
        norm = cfg_model["NORM"]
        norm_num_groups = cfg_model["NUM_GROUPS"]

        self.n_classes = cfg_model["N_CLASSES"]

        self.ini_conv = nn.Conv2d(in_channels,in_channels,1,1,0)

        self.encoder1 = customResBlock(in_channels=in_channels, out_channels=init_features, norm=norm, norm_num_groups=norm_num_groups)
        self.pool1 = customDownsample(init_features)
        self.encoder2 = customResBlock(in_channels=init_features, out_channels=init_features*2, norm=norm, norm_num_groups=norm_num_groups)
        self.pool2 = customDownsample(init_features*2)
        self.encoder3 = customResBlock(in_channels=init_features*2, out_channels=init_features*4, norm=norm, norm_num_groups=norm_num_groups)
        self.pool3 = customDownsample(init_features*4)
        self.encoder4 = customResBlock(in_channels=init_features*4, out_channels=init_features*8, norm=norm, norm_num_groups=norm_num_groups)
        self.bottleneck = customResBlock(in_channels=init_features*8, out_channels=init_features*8, norm=norm, norm_num_groups=norm_num_groups)

        self.decoder4 = customResBlock(in_channels=init_features*8, out_channels=init_features*8, norm=norm, norm_num_groups=norm_num_groups)
        self.upconv3 = customUpsample(init_features*8, init_features*4)
        self.decoder3 = customResBlock(in_channels=init_features*8, out_channels=init_features*4, norm=norm, norm_num_groups=norm_num_groups)
        self.upconv2 = customUpsample(init_features*4, init_features*2)
        self.decoder2 = customResBlock(in_channels=init_features*4, out_channels=init_features*2,norm=norm, norm_num_groups=norm_num_groups)
        self.upconv1 = customUpsample(init_features*2, init_features)
        self.decoder1 = customResBlock(in_channels=init_features*2, out_channels=init_features, norm=norm, norm_num_groups=norm_num_groups)

        self.conv = nn.Conv2d(in_channels=init_features, out_channels=self.n_classes, kernel_size=1, stride=1, padding=0)

        if self.n_classes==1:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x):
        x = self.ini_conv(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        logits = self.conv(dec1)
        return logits
    
    def get_bottleneck(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(enc4)
        return bottleneck
    
    def probs(self, x):
        if self.n_classes==1:
            return self(x)
        else:
            softmax = nn.Softmax(1)
            return softmax(self(x))
