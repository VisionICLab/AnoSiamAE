from torch import nn, Tensor
import torch

class ImageNetNorm(nn.Module):
    r"""Normalizes channels with respect to ImageNet's mean and standard deviation.

    References:
        | ImageNet: A large-scale hierarchical image database (Deng et al, 2009)
        | https://ieeexplore.ieee.org/document/5206848

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> normalize = ImageNetNorm()
        >>> x = normalize(x)
        >>> x.shape
        torch.Size([5, 3, 256, 256])
    """
    def __init__(self, device="cuda"):
        super().__init__()
        MEAN: Tensor = torch.tensor([0.485, 0.456, 0.406]).to(device)
        STD: Tensor = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self.register_buffer('shift', MEAN.reshape(3, 1, 1))
        self.register_buffer('scale', STD.reshape(3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.shift) / self.scale
    