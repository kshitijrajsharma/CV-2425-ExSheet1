import torch
from einops import rearrange

# Exercise 1
def assignment_ex1() -> torch.Tensor:
    T = torch.arange(16*3*32*32).float()
    T = T.view(16,3,32,32)
    return T

# Exercise 2
def assignment_ex2() -> torch.Tensor:
    T0 = torch.arange(16*3*3).view(16,3,3).float()
    T1 = T0*3.0    
    return torch.matmul(T0,T1).sum(dim=(1,2))

# Exercice 3
def assignment_ex3() -> torch.Tensor:
    T0 = torch.arange(16*3*3).view(16,3,3).float()
    return rearrange(T0, 'b i j -> b j i')