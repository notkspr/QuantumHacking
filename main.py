import torch.nn as nn
import torch
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# ignore complex warnings
warnings.filterwarnings("ignore")

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
            plt.plot(y[i])
            plt.title(f"Qubit {i}")
            plt.savefig(str(Path(__file__).parent.absolute())+f"/results/qubit/{i}.png")
            plt.clf()
        
        a = np.array(self.a.detach())
        for i in range(a.shape[0]):
            plt.plot(a[i])
            plt.title(f"A {i}")
            plt.savefig(str(Path(__file__).parent.absolute())+f"/results/a/{i}.png")
            plt.clf()


# Helper functions
def fidelity(current, target):
    # fidelity function: find the "difference" between the current and the target matrix/pulse
    return torch.abs(torch.trace(torch.matmul(current.adjoint(),target))/target.shape[0])**2

def penalty(matrix, weight):
    # penalty function: makes the pulse smoother
    return torch.sum(torch.diff(matrix, dim=1)**2)*weight

def train(model, optim, target, requiredaccuracy, maxiterations, weight):
    # training data
    loss = 1
    i = 0
    while loss >= requiredaccuracy and i<maxiterations:
        predictedOut = model()
        fid = fidelity(predictedOut, target)
        pen = penalty(model.a, weight)
        loss = fid + pen
        if i % 100 == 99:
            print(f"Epoch {i+1}:\n - loss: {loss.item()}\n - fidelity: {fid}\n - penalty: {pen}")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        i += 1
    print("Training Finished:")
    print(f"loss: {loss.item()}\n - fidelity: {fidelity(model(), target)}\n - penalty: {penalty(model.a, weight)}")

    
# deprecated
def trainn(Hc, dt, N, maxpower, n, model, optim, target, requiredaccuracy, maxiteraitons, maxmaxiteraitons):
    # train "n" -> choose best -> continue training
    # haven't finished debugging
    seed = []
    models = []
    losss = []
    for _ in range(n):
        seed.append(torch.randn(Hc.shape[0], N))
        models.append(model())
    
    for i in range(n):
        current = models[i](Hd, Hc, dt, N, maxpower, seed)
        train(current, optim, target, requiredaccuracy, maxiteraitons)
        losss.append(fidelity(current, target))
    
    bestindex = 0
    best = 0
    for i in range(n):
        if losss[i] < best:
            best = losss[i]
            bestindex = i
    train(models[bestindex], optim, target, requiredaccuracy, maxmaxiteraitons)

# Target Data
N = 500
dt = 10**(-4)

X = torch.tensor([[0, 1], [1, 0]])
Y = torch.tensor([[0, complex(0, -1)], [complex(0, 1), 0]])
Z = torch.tensor([[1, 0], [0, -1]])
I = torch.eye(2)

Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
H1c = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)
maxpower = 20000

# Target Gate (Toffoli)
H0 = torch.eye(8, dtype=torch.complex64)
H0[6][6] = 0
H0[6][7] = 1
H0[7][7] = 0
H0[7][6] = 1
Hc = torch.stack([H1c, H2c])

# Main Function
model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
adam = torch.optim.Adam(model.parameters(), lr = 0.001)
train(model=model, optim=adam, target=H0, requiredaccuracy=0.01, maxiterations=15000, weight=0.1)
model.plot()