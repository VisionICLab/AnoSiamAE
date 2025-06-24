from torch import nn
import torch

from torch import nn
import torch
from .encoder import Encoder
from .decoder import Decoder
from .blocks import customResBlock

class SiameseAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, num_channels, attention_levels, latent_channels, norm, norm_num_groups, dropout, dropout_input,final_attention, hidden_channel=256):
        super(SiameseAutoencoder, self).__init__()

        in_channels = in_channels
        self.encoder = Encoder(
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm=norm,
            norm_num_groups=norm_num_groups,
            attention_levels=attention_levels,
            dropout=dropout,
            dropout_input=dropout_input,
            final_attention=final_attention,
        )

        self.latent_predictor = nn.Sequential(nn.Conv2d(latent_channels,hidden_channel,3,1,1),
                                                nn.BatchNorm2d(hidden_channel),
                                                nn.SiLU(),
                                                nn.Conv2d(hidden_channel,latent_channels,3,1,1),
                                                )
        
        self.latent_projector = nn.Sequential(nn.Conv2d(latent_channels,hidden_channel,3,1,1),
                                                nn.BatchNorm2d(hidden_channel),
                                                nn.SiLU(),
                                                nn.Conv2d(hidden_channel,hidden_channel,3,1,1),
                                                nn.BatchNorm2d(hidden_channel),
                                                nn.SiLU(),
                                                nn.Conv2d(hidden_channel,latent_channels,3,1,1),
                                                )
        
        self.decoder = Decoder(
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm=norm,
            norm_num_groups=norm_num_groups,
            attention_levels=attention_levels,
            dropout=dropout,
            final_attention=final_attention,
        )
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z, mask=None):
        y = self.decoder(z, mask)
        if not mask is None:
            return self.sigmoid(y)*mask
        return self.sigmoid(y)
    
    def forward(self,x, x_aug, mask=None):
        z = self.encode(x)
        z_aug = self.encode(x_aug)

        _, zp = self.project(z)
        p_aug,_ = self.project(z_aug)

        y_aug = self.decode(z_aug,mask)

        return y_aug, zp, p_aug
    
    def project(self, z, mask=None):
        proj = self.latent_projector(z)
        pred = self.latent_predictor(proj)
        return pred, proj
    
    def output(self, x, mask=None):
        z = self.encode(x)

        p = self.latent_projector(z)

        y = self.decode(z,mask)

        return y, z, p

    def encode_stage_2_inputs(self, x):
        return self.encoder(x)

    def reconstruct(self,x, mask=None):
        z = self.encode(x)
        reconstruction = self.decode(z, mask)
        return reconstruction

    def decode_stage_2_outputs(self, z, mask=None):
        # z = self.latent_projector(z)
        image = self.decode(z, mask)
        return image

    def get_encoder_layers(self,x):
        return self.encoder.get_intermediate_layers(x)

def build_model(params, device = 'cuda'):
    model = SiameseAutoencoder(
            in_channels=params.IN_CHANNEL,
            out_channels=params.OUT_CHANNEL,
            num_channels=tuple(params.NUM_CHANNELS),
            latent_channels=params.LATENT_CHANNEL,
            num_res_blocks=params.NUM_RES_BLOCKS,
            norm=params.NORM,
            norm_num_groups=params.NUM_GROUPS,
            attention_levels=tuple(params.ATTENTION_LEVELS),
            dropout=params.DROPOUT,
            dropout_input=params.DROPOUT_INPUT,
            final_attention=params.FINAL_ATTENTION,
        ).to(device)
    return model