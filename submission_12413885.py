"""Submission for exercise sheet 1

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
from einops import rearrange


# Exercise 1.1
def assignment_ex1() -> torch.Tensor:
    return torch.arange(16 * 3 * 32 * 32, dtype=torch.float32).reshape(16, 3, 32, 32)


# Exercise 1.2
def assignment_ex2() -> torch.Tensor:
    t0 = torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3)
    return torch.sum(torch.matmul(t0, t0 * 3), dim=[1, 2])


# Exercice 1.3
def assignment_ex3() -> torch.Tensor:
    t0 = torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3)
    return rearrange(t0, "n w h -> n h w")
