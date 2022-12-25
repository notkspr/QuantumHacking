import torch.nn as nn
import torch
from functools import reduce
import matplotlib.pyplot as plt
import time
import math


class QOC(nn.Module):
    def __init__(self, Hd, Hc, dt, N, maxpower, seed):

        super().__init__()

        self.Hd = Hd
        self.Hc = Hc
        self.dt = dt
        self.N = N
        self.maxpower = maxpower
        self.seed = seed
        self.a = nn.Parameter(self.seed) # seed = torch.randn(Hc.shape[0], N))

    def forward(self):
        iH = 1j*self.Hd + torch.einsum("li, ljk -> ijk", 1j*self.a, self.Hc)*self.maxpower
        U = torch.linalg.matrix_exp(-iH*self.dt)
        return reduce(torch.matmul, U)

    def plot(self):
        xlist = torch.tensor([1]) + torch.tensor(range(self.a.shape[1]))
        ylist1 = self.a.detach()[0]
        ylist2 = self.a.detach()[1]
        plt.plot(xlist, ylist1)
        plt.plot(xlist, ylist2)
        plt.show()


def fidelity(target, current):
    return torch.abs(torch.trace(torch.matmul(current.adjoint(),target))/target.shape[0])**2

def penalty(model, weight):
    return torch.sum(torch.diff(model.a, dim=1)**2)*weight

def train(model, optim, target, requiredaccuracy, penaltyconst, weight, maxiterations):
    loss = 0
    i = 0
    while (1-fidelity(target, model()) >= requiredaccuracy or penalty(model, weight) > penaltyconst*weight) and i<maxiterations:
        loss = -fidelity(target, model()) + penalty(model, weight)
        if i % 100 == 0:
            if i == 0:
                start_time = time.time()
                timing = time.time() - start_time
                print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                print(f"- loss: {loss.item()}")
                print(f"- fidelity: {fidelity(target, model())}")
                print(f"- penalty: {penalty(model, weight)}\n")
            else:
                timing = time.time()-start_time
                print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
                print(f"- loss: {loss.item()}")
                print(f"- fidelity: {fidelity(target, model())}")
                print(f"- penalty: {penalty(model, weight)}")
                print(f"- average time per epoch: {(timing)/(i+1)} seconds\n")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        i += 1
    timing = time.time()-start_time
    print(f"Epoch {i} ({math.floor(timing/3600):02}:{math.floor(timing/60 % 60):02}:{math.floor(timing % 60):02}.{str(timing % 1)[2:]})")
    loss = -fidelity(target, model()) + penalty(model, weight)
    print(f"- loss: {loss.item()}")
    print(f"- fidelity: {fidelity(target, model())}")
    print(f"- penalty: {penalty(model, weight)}\n")
    print(f"Average time per epoch: {(timing)/(i+1)} seconds")
    model.plot()


n = 3
N = 1.25*(4**n)
'''
n = 2, N = 20
n = 3, N = 80
n = 4, N = 300
n = 5, N = 1200
n = 6, N = 5000
'''
dt = 0.05

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
I = torch.eye(2, dtype=torch.complex64)

# For defining n-qubit drift & control Hamiltonian
def hat(gate, index):
    hatgate = torch.tensor([1], dtype=torch.complex64)
    for i in range(n):
        if i != index:
            unitary = I
        if i == index:
            unitary = gate
        hatgate = torch.kron(hatgate, unitary)
    return hatgate

Xs = torch.stack(list(hat(X, i) for i in range(n)))
Ys = torch.stack(list(hat(Y, i) for i in range(n)))
Zs = torch.stack(list(hat(Z, i) for i in range(n)))

F = torch.zeros(n, n, dtype=torch.complex64)/2
for i in range(n):
    for j in range(n):
        if j == i + 1:
            F[i][j] = 1

Hd = torch.einsum("ij, ikl, jlm -> km", F, Zs, Zs)
Hc = torch.stack([torch.einsum("ijk -> jk", Xs), torch.einsum("ijk -> jk", Ys)])
maxpower = 1

# Target unitary
H0 = torch.eye(2**n, dtype=torch.complex64)

model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
train(model = model, optim = optim, target = H0, requiredaccuracy = 0.01, penaltyconst = 10, weight = 1/N, maxiterations = 100000)
