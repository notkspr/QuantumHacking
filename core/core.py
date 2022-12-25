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
        H = 1j*self.Hd + torch.einsum("li, ljk -> ijk", 1j*self.a, self.Hc)*self.maxpower
        U = torch.linalg.matrix_exp(-H*self.dt)
        return reduce(torch.matmul, U)

    def plot(self):
        y = np.array(self.forward().detach())
        
        for i in range(y.shape[0]):
            try:    
                open(str(Path(__file__).parent.absolute())+f"/results/qubit/{i}.png", "x").close()
            except:
                pass
            plt.plot(y[i])
            plt.title(f"Qubit {i}")
            plt.savefig(str(Path(__file__).parent.absolute())+f"/results/qubit/{i}.png")
            plt.clf()
        
        a = np.array(self.a.detach())
        for i in range(a.shape[0]):
            try:
                open(str(Path(__file__).parent.absolute())+f"/results/a/{i}.png", "x").close()
            except:
                pass
            plt.plot(a[i])
            plt.title(f"A {i}")
            plt.savefig(str(Path(__file__).parent.absolute())+f"/results/a/{i}.png")
            plt.clf()
    def print(self):
        print(self.a)


# Helper functions
def fidelity(current, target):
    # fidelity function: find the "difference" between the current and the target matrix/pulse
    return -torch.abs(torch.trace(torch.matmul(current.adjoint(),target))/target.shape[0])**2

def penalty(matrix, weight):
    # penalty function: makes the pulse smoother
    return torch.sum(torch.diff(matrix, dim=1)**2)*weight

def train(model, optim, target, accuracy, smoothness, maxiterations, weight):
    # training data
    fid = 0
    pen = 1
    loss = 0
    i = 0
    while (1+fid >= accuracy or pen >= smoothness*weight) and i<maxiterations:
        predictedOut = model()
        fid = fidelity(predictedOut, target)
        pen = penalty(model.a, weight)
        loss = (1+fid) + pen
        if i % 100 == 99:
            print(f"Epoch {i+1}:\n - loss: {loss.item()}\n - fidelity: {fid}\n - smoothness: {pen}")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        i += 1
    print("Training Finished:")
    print(f" - loss: {loss.item()}\n - fidelity: {fidelity(model(), target)}\n - smoothness: {penalty(model.a, weight)}")

def trainwithtiming(model, optim, target, requiredaccuracy, penaltyconst, weight, maxiterations):
    loss = 0
    i = 0
    while (1+fidelity(target, model()) >= requiredaccuracy or penalty(model.a, weight) > penaltyconst*weight) and i<maxiterations:
        loss = 1+fidelity(target, model()) + penalty(model.a, weight)
        if i % 100 == 0:
            if i == 0:
                start_time = time.time()
                timing = time.time() - start_time
                print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                print(f"- loss: {loss.item()}")
                print(f"- fidelity: {-fidelity(target, model())}")
                print(f"- penalty: {penalty(model.a, weight)}\n")
            else:
                timing = time.time()-start_time
                print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                print(f"- loss: {loss.item()}")
                print(f"- fidelity: {-fidelity(target, model())}")
                print(f"- penalty: {penalty(model.a, weight)}")
                print(f"- average time per epoch: {(timing)/(i+1)} seconds\n")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        i += 1
    timing = time.time()-start_time
    print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
    loss = -fidelity(target, model()) + penalty(model.a, weight)
    print(f"- loss: {loss.item()}")
    print(f"- fidelity: {-fidelity(target, model())}")
    print(f"- penalty: {penalty(model.a, weight)}\n")
    print(f"Average time per epoch: {(timing)/(i+1)} seconds")
    model.plot()
