import torch
import torch.nn as nn


class customResBlock(nn.Module):
    def __init__(self, in_channels, norm, norm_num_groups, out_channels, kernel=[3,3], stride=[1,1], padding=[1,1], bias=False):
        super().__init__()
        bias = True
        if norm=="batch":
            bias=False
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        if norm == "batch":
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == "group":
            self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels)
        elif norm == "layer":
            self.norm1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        else:
            self.norm1=nn.Identity()
            self.norm2=nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel[0], stride=stride[0], padding=padding[0], bias=bias)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=kernel[1], stride=stride[1], padding=padding[1])

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        else:
            self.nin_shortcut = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h+x
    
    def get_activations(self,x:torch.Tensor):
        l=[]
        h = self.norm1(x)
        h = self.act(h)
        l.append(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        l.append(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h+x, l
        
class customAttention(nn.Module):
    def __init__(self, num_channels, num_heads=1, norm="group", norm_num_groups=32):
        """
            Inspired by https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
            and https://arxiv.org/pdf/1706.03762
        """
        super().__init__()
        self.query = nn.Conv2d(in_channels=num_channels, out_channels = num_channels//8, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=num_channels, out_channels = num_channels//8, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=num_channels, out_channels = num_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) 

    def forward(self, x):
        batch, channel, height, width = x.shape
        query = self.query(x).view(batch, -1, height*width).transpose(1,2) # [B,HxW,C//8]
        key = self.key(x).view(batch, -1, height*width) # [B,C//8,HxW]

        logits = torch.bmm(query,key) # [B,HxW,HxW]
        attention = self.softmax(logits) # [B,HxW,HxW]

        value = self.value(x).view(batch, -1, height*width) # [B,C,HxW]
        output = torch.bmm(attention,value.transpose(1,2)).transpose(1,2) 
        output = output.view(batch, channel, height,width) 

        output = self.gamma*output + x
        return output, attention
    
class customDownsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class customUpsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.upsamle = nn.Upsample(scale_factor = 2, mode = 'nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.upsamle(x)
        x = self.conv(x)
        return x
