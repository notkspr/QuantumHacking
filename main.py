import torch.nn as nn
import torch

class QOC(nn.Module):
    def __init__(self, Hd, Hc, dt, N):

        super().__init__()

        self.Hd = Hd
        self.Hc = Hc
        self.dt = dt
        self.N = N

        self.a1 = nn.Parameter(torch.rand(N).requires_grad_())
        self.a2 = nn.Parameter(torch.rand(N).requires_grad_())

        self.linear = nn.Linear(in_features=N, out_features=N)


    def forward(self):
        pass

def fidelity(target, current):
    return -(abs(torch.trace(torch.matmul(torch.adjoint(current),target))/8))**2

def train():
    pass



