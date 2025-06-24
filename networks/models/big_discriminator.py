import torch.nn as nn
from .blocks import customResBlock, customDownsample

class big_discriminator(nn.Module):

    def __init__(self, in_shape=(3,256,256), num_channels = (64, 128, 256, 512, 1), num_res_blocks=1, dropout=0.0, num_groups=8, norm="batch", patch_disc=False):
        super(big_discriminator, self).__init__()
        self.patch_disc = patch_disc

        in_channels = in_shape[0]
        out_channel, w, h = in_shape

        self.layers = nn.Sequential()
        self.layers.add_module("conv0",nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=3, stride=1, padding=1))
        for i,c in enumerate(num_channels):
            in_channel = out_channel
            out_channel = c
            for j in range(num_res_blocks):
                self.layers.add_module("ResBlock{}_{}".format(i+1,j),customResBlock(in_channels=in_channel,norm=norm,norm_num_groups=num_groups,out_channels=out_channel, kernel=[5,3], stride=[1,1], padding=[2,1]))
                in_channel = out_channel    
            self.layers.add_module("drop{}".format(i+1), nn.Dropout(dropout))
            if i<len(num_channels)-1:
                self.layers.add_module("ds{}".format(i+1), customDownsample(in_channels=in_channel))
                w, h = w//2, h//2 
        self.layers.add_module("conv{}".format(i+2),nn.Conv2d(in_channels=out_channel, out_channels=1, kernel_size=4, stride=1, padding=1))
        w, h = w-1, h-1
        if not patch_disc:
            self.linear = nn.Linear(1*w*h,1)
        self.apply(self.init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if not self.patch_disc:
            x = self.linear(x.view(x.shape[0],-1))
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
    
    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad
        
    def get_intermediate_layers(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, customResBlock):
                features.append(x)
        return features


def build_model(params,device="cuda"):
    model = big_discriminator(in_shape= params.IN_SHAPE,num_channels = params.NUM_CHANNELS, dropout=params.DROPOUT, num_res_blocks=params.NUM_RES_BLOCKS,
                              num_groups=params.NUM_GROUPS, norm=params.NORM, patch_disc=params.PATCH).to(device)
    return model
