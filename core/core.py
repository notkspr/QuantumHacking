import torch.nn as nn
import torch
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time


# Custom neural network module
class QOC(nn.Module):
    def __init__(self, Hd, Hc, dt, N, maxpower, seed):

        super().__init__()

        self.Hd = Hd
        self.Hc = Hc
        self.dt = dt
        self.N = N
        self.maxpower = maxpower
        self.a = nn.Parameter(seed) # torch.randn(Hc.shape[0], N))


    def forward(self):
        iH = 1j*self.Hd + torch.einsum("li, ljk -> ijk", 1j*self.a, self.Hc)*self.maxpower
        U = torch.linalg.matrix_exp(-iH*self.dt)
        return reduce(torch.matmul, U)

    def plot(self):
        a = np.array(self.a.detach())
        for i in range(a.shape[0]):
            try:
                open(str(Path(__file__).parent.absolute())+f"/results/a/a{i+1}.png", "x").close()
            except:
                pass
            plt.plot(np.linspace(0, self.dt*self.N, self.N), a[i])
            plt.title(f"Control Pulse {i+1}")
            plt.xlabel("time evolved (s)")
            plt.ylabel("pulse amplitude")
            plt.savefig(str(Path(__file__).parent.absolute())+f"/results/a/a{i+1}.png")
            plt.clf()

    def printa(self):
        print(self.a)


# Helper functions
def fidelity(current, target):
    # fidelity function: find the "difference" between the current and the target matrix/pulse
    return torch.abs(torch.trace(torch.matmul(current.adjoint(),target))/target.shape[0])**2

def penalty(matrix, weight):
    # penalty function: makes the pulse smoother
    return torch.sum(torch.diff(matrix, dim=1)**2)*weight


# Training function
def train(model, optim, target, accuracy, roughness, weight, maxiterations, benchbool):
    fid = fidelity(target, model())
    pen = penalty(model.a, weight)
    loss = 1 - fid + pen
    i = 0
    while ((1-fid) >= accuracy or pen > roughness*weight) and i < maxiterations:
        if i % 100 == 0:
            if i == 0:
                start_time = time.time()
                timing = time.time() - start_time
                if benchbool != True:
                    print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                    print(f"- Loss: {loss.item()}\n- Fidelity: {fid}\n- Roughness: {pen}\n")
            else:
                timing = time.time() - start_time
                if benchbool != True:
                    print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                    print(f"- Loss: {loss.item()}\n- Fidelity: {fid}\n- Roughness: {pen}\n- Average time per epoch: {(timing)/(i+1)} seconds\n")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        fid = fidelity(target, model())
        pen = penalty(model.a, weight)
        loss = 1 - fid + pen
        i += 1
    timing = time.time() - start_time
    fid = fidelity(target, model())
    pen = penalty(model.a, weight)
    loss = 1 - fid + pen
    if benchbool != True:
        print("Training Finished:")
        print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
        print(f"- Loss: {loss.item()}\n- Fidelity: {fid}\n- Roughness: {pen}\n- Average time per epoch: {(timing)/(i+1)} seconds\n")
        model.plot()
    itertime = (timing)/(i+1)
    if benchbool == True:
        return itertime
