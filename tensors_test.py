import torch

import tensors

def test_ex1():
    T = torch.load('assets/ex1_T.pt', weights_only=False)
    assert (T-tensors.assignment_ex1()).norm() < 1e-6
    
def test_ex2():
    A = torch.load('assets/ex2_T.pt', weights_only=False)
    assert (A-tensors.assignment_ex2()).norm() < 1e-6

def test_ex3():
    A = torch.load('assets/ex3_T.pt', weights_only=False)
    assert (A-tensors.assignment_ex3()).norm() < 1e-6  