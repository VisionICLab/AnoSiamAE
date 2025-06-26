import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .build import ARCH_REGISTRY

@ARCH_REGISTRY.register("autoencoder")
class AutoEncoder(nn.Module):
    def __init__(self, cfg_model:dict):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(cfg_model)
        self.decoder = Decoder(cfg_model)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        z = self.encoder(x)
        return z

    def reconstruct(self, x, mask=None):
        z = self.encode(x)
        reconstruction = self.decode(z, mask)
        return reconstruction

    def decode(self, z, mask=None):
        dec = self.decoder(z)
        if not mask is None:
            return self.sigmoid(dec)*mask
        return self.sigmoid(dec)

    def forward(self, x, mask=None):
        z = self.encode(x)
        reconstruction = self.decode(z, mask)
        return reconstruction

    def encode_stage_2_inputs(self, x):
        z = self.encode(x)
        return z

    def decode_stage_2_outputs(self, z, mask=None):
        out = self.decode(z, mask)
        return out

    def get_encoder_layers(self,x):
        return self.encoder.get_intermediate_layers(x)
