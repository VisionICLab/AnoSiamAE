import torch

def l1(x: torch.Tensor,dim: int,keepdim: bool = False,) -> torch.Tensor:
    x = x.abs()
    x = x.sum(dim=dim, keepdim=keepdim)
    return x

def l2(x: torch.Tensor,dim: int,keepdim: bool = False) -> torch.Tensor:
    x = x.square()
    x = x.sum(dim=dim, keepdim=keepdim)
    x = x.sqrt()
    return x