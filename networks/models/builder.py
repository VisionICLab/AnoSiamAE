import os
import torch
from . import discriminator, \
                spatial_autoencoder, spatial_vae,\
                siamese_autoencoder,\
                big_discriminator, vae_decoder, vae_encoder,\
                siamese, decoder, encoder, unet, draem, ganomaly


def build_ganomaly(params, device="cuda"):
    print("Building {} model".format('GANomaly Model'))
    model = ganomaly.build_model(params, device)
    return model

def build_draem(params, device="cuda"):
    print("Building {} model".format('DRAEM Model'))
    model = draem.build_model(params, device)
    return model

## Siamese autoencoder

def build_siamese(params, device="cuda"):
    print("Building {} model".format('SimSiam Model'))
    model = siamese.build_model(params, device)
    return model

def build_siamese_autoencoder(params, device="cuda"):
    print("Building {} model".format('Siamese Autoencoder'))
    model = siamese_autoencoder.build_model(params, device)
    return model


## Spatial Autoencoder Builder
## Architecture inspired from Baur et. al. AnoVAEGAN

def build_encoder(params,device="cuda"):
    print("Building {} model".format(" Spatial Encoder "))
    model = encoder.build_model(params, device)
    return model


def build_decoder(params,device="cuda"):
    print("Building {} model".format(" Spatial Decoder "))
    model = decoder.build_model(params, device)
    return model

def build_spatial_autoencoder(params,device="cuda"):
    print("Building {} model".format(" Spatial AutoEncoder"))
    model = spatial_autoencoder.build_model(params).to(device)
    return model

def build_unet(params, device="cuda"):
    print("Building {} model".format(" UNet"))
    model = unet.build_model(params).to(device)
    return model


## Discriminator Builder
## Architecture inspired from WGAN

def build_discriminator(params,device="cuda"):
    print("Building {} model".format("Batch Discriminator"))
    model = discriminator.build_model(params).to(device)
    return model

def build_big_discriminator(params,device="cuda"):
    print("Building {} model".format("Big Discriminator"))
    model = big_discriminator.build_model(params).to(device)
    return model

## VAE

def build_vae_encoder(params,device="cuda"):
    print("Building {} model".format(" VAE Encoder only"))
    model = vae_encoder.build_model(params).to(device)
    return model

def build_vae_decoder(params,device="cuda"):
    print("Building {} model".format(" VAE Decoder only"))
    model = vae_decoder.build_model(params).to(device)
    return model

def build_spatial_vae(params,device="cuda"):
    print("Building {} model".format("Spatial VAE"))
    model = spatial_vae.build_model(params).to(device)
    return model