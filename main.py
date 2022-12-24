import torch.nn as nn
import torch
from functools import reduce



class QOC(nn.Module):
    def __init__(self, Hd, Hc, dt, N, maxpower, seed):

        super().__init__()

        self.Hd = Hd
        self.Hc = Hc
        self.dt = dt
        self.N = N
        self.maxpower = maxpower
        self.a = nn.Parameter(seed) # seed = torch.randn(Hc.shape[0], N))


    def forward(self):
        H = 1j*self.Hd + torch.einsum("li, ljk -> ijk", 1j*self.a, self.Hc)*self.maxpower
        U = torch.linalg.matrix_exp(-H*self.dt)
        return reduce(torch.matmul, U)

    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt

        y = np.array(self.a.detach())
        print(y)
        
        for i in range(self.a.shape[0]):
            plt.plot(y[i])
            plt.show()
        

        

def fidelity(target, current):
    return -torch.abs(torch.trace(torch.matmul(current.adjoint(),target))/target.shape[0])**2

def penalty(model, weight):
    return torch.sum(torch.diff(model.a, dim=1)**2)*weight

def train(model, optim, target, requiredaccuracy, maxiterations):
    # training data
    loss = 0
    i = 0
    while 1+loss >= requiredaccuracy and i<maxiterations:
        predictedOut = model()
        loss = fidelity(predictedOut, target) #TODO:+penalty(model, weight)
        if i % 100 == 99:
            print(loss.item())
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        i += 1

    
# TODO
def trainn(Hc, dt, N, maxpower, n, model, optim, target, requiredaccuracy, maxiteraitons, maxmaxiteraitons):
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


H0 = torch.eye(8, dtype=torch.complex64)
H0[6][6] = 0
H0[6][7] = 1
H0[7][7] = 0
H0[7][6] = 1
Hc = torch.stack([H1c, H2c])
maxpower = 20000


model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))

adam = torch.optim.Adam(model.parameters(), lr = 0.001)
train(model, adam, H0, requiredaccuracy=0.01, maxiterations=10000)

model.plot()

#trainn(Hc, dt, N, maxpower, 100, QOC, adam, H0, 0.01, 500, 1000000)
