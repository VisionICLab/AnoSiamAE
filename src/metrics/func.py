from torch import Tensor
from typing import *
import torch
import torch.nn.functional as F

def gaussian_kernel(
    size: int,
    sigma: float = 1.0,
) -> Tensor:

    kernel = torch.arange(size, dtype=torch.float)
    kernel -= (size - 1) / 2
    kernel = kernel ** 2 / (2 * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel

def reduce_tensor(x: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()

    return x

def kernel_views(kernel: Tensor, n: int = 2) -> List[Tensor]:

    if n == 1:
        return [kernel]
    elif n == 2:
        return [kernel.unsqueeze(-1), kernel.unsqueeze(-2)]

    # elif n > 2:
    c, _, k = kernel.shape

    shape: List[int] = [c, 1] + [1] * n
    views = []

    for i in range(2, n + 2):
        shape[i] = k
        views.append(kernel.reshape(shape))
        shape[i] = 1

    return views

def channel_conv(
    x: Tensor,
    kernel: Tensor,
    padding: int = 0,
) -> Tensor:

    D = kernel.dim() - 2

    assert D <= 3, "PyTorch only supports 1D, 2D or 3D convolutions."

    if D == 3:
        return F.conv3d(x, kernel, padding=padding, groups=x.shape[-4])
    elif D == 2:
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[-3])
    elif D == 1:
        return F.conv1d(x, kernel, padding=padding, groups=x.shape[-2])
    else:
        return F.linear(x, kernel.expand(x.shape[-1]))


def channel_convs(
    x: Tensor,
    kernels: List[Tensor],
    padding: int = 0,
) -> Tensor:
    if padding > 0:
        pad = (padding,) * (2 * x.dim() - 4)
        x = F.pad(x, pad=pad)

    for k in kernels:
        x = channel_conv(x, k)

    return x