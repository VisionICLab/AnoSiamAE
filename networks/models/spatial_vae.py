from torch import nn
import torch
from networks.models.encoder import Encoder
from networks.models.decoder import Decoder
    
class Vae(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, num_channels, attention_levels, latent_channels, norm, norm_num_groups, dropout, dropout_input, final_attention):
        super().__init__()

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
        self.quant_conv_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=1 ,stride=1 , padding=0)
        self.quant_conv_log_sigma = nn.Conv2d(latent_channels, latent_channels, kernel_size=1 ,stride=1 , padding=0)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1 ,stride=1 , padding=0)

        self.latent_channels = latent_channels
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        return z_mu, z_log_var

    def sampling(self, z_mu, z_log_var):
        z_sigma = torch.exp(z_log_var / 2)
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def decode(self, z, mask=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, mask)
        if not mask is None:
            return self.sigmoid(dec)*mask
        return self.sigmoid(dec)
    
    def reconstruct(self, x, mask=None):
        z_mu, z_logvar = self.encode(x)
        reconstruction = 0
        for _ in range(10):
            reconstruction += self.decode(self.sampling(z_mu, z_logvar))
        return reconstruction/10

    def forward(self, x, mask=None):
        z_mu, z_logvar = self.encode(x)
        z = self.sampling(z_mu, z_logvar)
        reconstruction = self.decode(z, mask)
        return reconstruction, z_mu, z_logvar

    def encode_stage_2_inputs(self, x):
        z_mu, _ = self.encode(x)
        return z_mu

    def encode_stage_2_inputs(self, x):
        z_mu, z_logvar = self.encode(x)
        return self.sampling(z_mu, z_logvar)

    def get_encoder_layers(self,x):
        return self.encoder.get_intermediate_layers(x)


def build_model(params, device = 'cuda'):
    model = Vae(
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
