import torch
from torch import nn
from networks.models.blocks import customResBlock, customAttention, customDownsample

class VaeEncoder(nn.Module):
    def __init__(self, in_channels, num_channels, out_channels, num_res_blocks, norm, norm_num_groups, attention_levels, dropout, dropout_input,final_attention):
        super().__init__()
        assert len(num_channels)==len(attention_levels), "Not the same size for num_channels and attention_levels in encoder params"
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks for _ in range(len(num_channels))]
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.attention_levels = attention_levels
        self.final_attention = final_attention
        self.dropout = dropout
        self.dropout0 = nn.Dropout(dropout_input)
        blocks = []
        

        # Initial convolution
        blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=num_channels[0], kernel_size=3, stride=1, padding=1))
        blocks.append(nn.Dropout(dropout))
        

        # Residual and downsampling blocks
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]

            for _ in range(self.num_res_blocks[i]):
                blocks.append(customResBlock(in_channels=input_channel,norm=norm,norm_num_groups=norm_num_groups,out_channels=output_channel))
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(customAttention(num_channels=input_channel,norm=norm,norm_num_groups=norm_num_groups))
                blocks.append(nn.Dropout(dropout))
            if i<len(num_channels)-1:
                blocks.append(customDownsample(in_channels=input_channel))

        # Non-local attention block
        if self.final_attention:    
            blocks.append(customResBlock(in_channels=output_channel,norm=norm,norm_num_groups=norm_num_groups,out_channels=output_channel))
            blocks.append(customAttention(num_channels=output_channel,norm_num_groups=norm_num_groups))
            blocks.append(customResBlock(in_channels=output_channel,norm=norm,norm_num_groups=norm_num_groups,out_channels=output_channel))
            blocks.append(nn.Dropout(dropout))
        # Normalise and convert to latent size
        if norm=="batch":
            blocks.append(nn.BatchNorm2d(output_channel))
        elif norm=="group":
            blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=output_channel))
        elif norm=="layer":
            blocks.append(nn.GroupNorm(num_groups=output_channel, num_channels=output_channel))
        # blocks.append(nn.LeakyReLU())
        blocks.append(nn.Conv2d(output_channel, out_channels, 3,1,1))
        self.blocks = nn.ModuleList(blocks)

        self.quant_conv_mu = nn.Conv2d(out_channels, out_channels, kernel_size=1 ,stride=1 , padding=0)
        self.quant_conv_log_sigma = nn.Conv2d(out_channels, out_channels, kernel_size=1 ,stride=1 , padding=0)


    def sampling(self, z_mu, z_sigma):
            eps = torch.randn_like(z_sigma)
            z_vae = z_mu + eps * z_sigma
            return z_vae
    
    def forward(self, x):
        x = self.dropout0(x)
        for block in self.blocks:
            x = block(x)

        z_mu = self.quant_conv_mu(x)
        z_log_var = self.quant_conv_log_sigma(x)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)
        z = self.sampling(z_mu, z_sigma)
        return z, z_mu, z_sigma
    
    def get_intermediate_layers(self, x):
        L=[]
        for block in self.blocks:
            x=block(x)
            if isinstance(block, nn.Conv2d) or isinstance(block,customResBlock):
                L.append(x)
        return L

def build_model(params, device = 'cuda'):
    model = VaeEncoder(
            in_channels=params.IN_CHANNEL,
            out_channels=params.LATENT_CHANNEL,
            num_channels=tuple(params.NUM_CHANNELS),
            num_res_blocks=params.NUM_RES_BLOCKS,
            norm=params.NORM,
            norm_num_groups=params.NUM_GROUPS,
            attention_levels=tuple(params.ATTENTION_LEVELS),
            dropout=params.DROPOUT,
            dropout_input=params.DROPOUT_INPUT,
            final_attention=params.FINAL_ATTENTION,
        ).to(device) 
    return model


