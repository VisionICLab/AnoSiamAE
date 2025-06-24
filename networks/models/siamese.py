import torch.nn as nn
from .encoder import Encoder


class Siamese(nn.Module):
    def __init__(self, in_shape, num_res_blocks, num_channels, pred_dim, attention_levels, latent_channels, norm, norm_num_groups, dropout, dropout_input,final_attention):
        super(Siamese, self).__init__()
        in_channels, w, h = in_shape
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

        self.latent_projector = nn.Sequential(nn.BatchNorm2d(latent_channels),
                                              nn.LeakyReLU(inplace=True),
                                              nn.Conv2d(in_channels=latent_channels, out_channels=1, kernel_size=5, stride=2, padding=2),
                                              nn.BatchNorm2d(1),
                                              nn.LeakyReLU(inplace=True),
                                              nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
                                              nn.Upsample(scale_factor = 2, mode = 'nearest'),
                                              nn.Conv2d(in_channels=1, out_channels=latent_channels, kernel_size=5, stride=1, padding=2),
                                              )

    def encode(self, x):
        z = self.encoder(x)
        return z


    def forward(self,x, x_aug):
        z = self.encode(x)
        z_aug = self.encode(x_aug)

        p = self.latent_projector(z)
        p_aug = self.latent_projector(z_aug)

        return z, p, z_aug, p_aug
    
    def output(self, x):
        z = self.encode(x)
        p = self.latent_projector(z)

        return z, p

    def encode_stage_2_inputs(self, x):
        return self.encoder(x)

    def get_encoder_layers(self,x):
        return self.encoder.get_intermediate_layers(x)

def build_model(params, device = 'cuda'):
    model = Siamese(
            in_shape=params.IN_SHAPE,
            num_channels=tuple(params.NUM_CHANNELS),
            latent_channels=params.LATENT_CHANNEL,
            pred_dim=params.PRED_DIM,
            num_res_blocks=params.NUM_RES_BLOCKS,
            norm=params.NORM,
            norm_num_groups=params.NUM_GROUPS,
            attention_levels=tuple(params.ATTENTION_LEVELS),
            dropout=params.DROPOUT,
            dropout_input=params.DROPOUT_INPUT,
            final_attention=params.FINAL_ATTENTION,
        ).to(device)
    return model