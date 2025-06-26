from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .build import ARCH_REGISTRY

@ARCH_REGISTRY.register("siamese_autoencoder")
class SiameseAutoencoder(nn.Module):
    def __init__(self, cfg_model:dict):
        super(SiameseAutoencoder, self).__init__()

        self.encoder = Encoder(cfg_model)
        latent_channels = cfg_model["LATENT_CHANNEL"]
        hidden_channel = cfg_model["HIDDEN_CHANNEL"]

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
        
        self.decoder = Decoder(cfg_model)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z, mask=None):
        y = self.decoder(z)
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
