from torch import nn
from networks.models.blocks import customResBlock, customAttention, customDownsample

class Encoder(nn.Module):
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
            # blocks.append(customResBlock(in_channels=output_channel,norm=norm,norm_num_groups=norm_num_groups,out_channels=output_channel))
            blocks.append(customAttention(num_channels=output_channel,norm_num_groups=norm_num_groups))
            # blocks.append(customResBlock(in_channels=output_channel,norm=norm,norm_num_groups=norm_num_groups,out_channels=output_channel))
        # Normalise and convert to latent size
        if norm=="batch":
            blocks.append(nn.BatchNorm2d(output_channel))
        elif norm=="group":
            blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=output_channel))
        elif norm=="layer":
            blocks.append(nn.GroupNorm(num_groups=output_channel, num_channels=output_channel))
        blocks.append(nn.Conv2d(output_channel, out_channels, 3,1,1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.dropout0(x)
        for block in self.blocks:
            if isinstance(block, customAttention):
                x,_ = block(x)
            else:
                x = block(x)
        return x
    
    def get_intermediate_layers(self, x):
        L=[]
        for block in self.blocks:
            if isinstance(block, customAttention):
                x,_ = block(x)
            elif isinstance(block,customResBlock):
                x,l = block.get_activations(x)
                L+=l
            else:
                x=block(x)
        return L
    
def build_model(params, device = 'cuda'):
    model = Encoder(
            in_channels=params.IN_CHANNEL,
            num_channels=tuple(params.NUM_CHANNELS),
            out_channels=params.LATENT_CHANNEL,
            num_res_blocks=params.NUM_RES_BLOCKS,
            norm=params.NORM,
            norm_num_groups=params.NUM_GROUPS,
            attention_levels=tuple(params.ATTENTION_LEVELS),
            dropout=params.DROPOUT,
            dropout_input=params.DROPOUT_INPUT,
            final_attention=params.FINAL_ATTENTION,
        ).to(device) 
    return model