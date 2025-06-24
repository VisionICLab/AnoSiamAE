import torch.nn as nn
from networks.models.encoder import Encoder
from networks.models.decoder import Decoder
from networks.models.unet import UNet
import torch
import random 

class SpatialAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, num_channels, attention_levels, latent_channels, norm, norm_num_groups, dropout, dropout_input,final_attention):
        super(SpatialAutoencoder, self).__init__()

        self.encoder1 = Encoder(
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
        self.encoder2 = Encoder(
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
        self.latent_channels = latent_channels
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        z = self.encoder1(x)
        return z

    def reconstruct(self, x, mask=None):
        z = self.encode(x)
        reconstruction = self.decode(z, mask)
        return reconstruction

    def decode(self, z, mask=None):
        dec = self.decoder(z, mask)
        if not mask is None:
            return self.sigmoid(dec)*mask
        return self.sigmoid(dec)

    def forward(self, x, mask=None):
        z = self.encode(x)
        reconstruction = self.decode(z, mask)
        z_enc = self.encoder2(reconstruction)
        return reconstruction, z, z_enc

    def encode_stage_2_inputs(self, x):
        z = self.encode(x)
        return z

    def decode_stage_2_outputs(self, z, mask=None):
        out = self.decode(z, mask)
        return out

    def get_encoder_layers(self,x):
        return self.encoder.get_intermediate_layers(x)



def build_model(params, device = 'cuda'):
    model = SpatialAutoencoder(
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
