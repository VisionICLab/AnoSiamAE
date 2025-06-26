from torch import nn
from .layers.blocks import customResBlock, customAttention, customUpsample
from torchvision.transforms import v2 as transforms
from .build import ARCH_REGISTRY

@ARCH_REGISTRY.register("decoder")
class Decoder(nn.Module):
    def __init__(self, cfg_model:dict):
        super().__init__()
        num_channels = cfg_model["NUM_CHANNELS"]
        in_channels = cfg_model["LATENT_CHANNEL"]
        out_channels = cfg_model["OUT_CHANNEL"]
        num_res_blocks = cfg_model["NUM_RES_BLOCKS"]
        norm = cfg_model["NORM"]
        norm_num_groups = cfg_model["NUM_GROUPS"]
        attention_levels = cfg_model["ATTENTION_LEVELS"]
        final_attention = cfg_model["FINAL_ATTENTION"]
        dropout = cfg_model["DROPOUT"]

        assert len(num_channels)==len(attention_levels), "Not the same size for num_channels and attention_levels in decoder params"

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks for _ in range(len(num_channels))]

        reversed_block_out_channels = list(reversed(num_channels))
        # reversed_block_out_channels.append(out_channels)

        blocks = []
        # Initial convolution
        blocks.append(nn.Conv2d(in_channels, reversed_block_out_channels[0], kernel_size=3, stride=1, padding=1))

        # Non-local attention block
        if final_attention:
            blocks.append(customResBlock(in_channels=reversed_block_out_channels[0],norm=norm,norm_num_groups=norm_num_groups,out_channels=reversed_block_out_channels[0]))
            blocks.append(customAttention(num_channels=reversed_block_out_channels[0],norm_num_groups=norm_num_groups))
            blocks.append(customResBlock(in_channels=reversed_block_out_channels[0],norm=norm,norm_num_groups=norm_num_groups,out_channels=reversed_block_out_channels[0]))

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(1,len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            for _ in range(reversed_num_res_blocks[i-1]):
                blocks.append(customResBlock(in_channels=block_in_ch,norm = norm,norm_num_groups=norm_num_groups,out_channels=block_out_ch))
                block_in_ch = block_out_ch
                blocks.append(nn.Dropout(dropout))

                if reversed_attention_levels[i-1]:
                    blocks.append(customAttention(num_channels=block_out_ch,norm=norm,norm_num_groups=norm_num_groups))
            if i<len(num_channels):
                blocks.append(customUpsample(in_channels=block_out_ch))

        # blocks.append(customResBlock(in_channels=block_out_ch,norm = norm,norm_num_groups=norm_num_groups,out_channels=block_out_ch))
        if norm=="batch":
            blocks.append(nn.BatchNorm2d(block_out_ch))
        elif norm=="group":
            blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_ch))
        elif norm=="layer":
            blocks.append(nn.GroupNorm(num_groups=block_out_ch, num_channels=block_out_ch))
        blocks.append(nn.Conv2d(block_out_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            if isinstance(block, customAttention):
                x, _ = block(x)
            else:
                x = block(x)
        return x
    