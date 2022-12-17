import torch.nn as nn
import torch

class QOC(nn.Module):
    def __init__(self, Hd, H1c, H2c, dt, N):

        super().__init__()

        self.Hd = Hd
        self.H1c = H1c
        self.H2c = H2c
        self.dt = dt
        self.N = N
        self.a1 = nn.Parameter(torch.randn(N))
        self.a2 = nn.Parameter(torch.randn(N))


    def forward(self):
        output = torch.complex(torch.eye(8), torch.zeros(8))
        for i in range(self.N):
            output = torch.matmul(torch.linalg.matrix_exp(complex(0, -1)*(self.Hd + self.a1[i]*self.H1c + self.a2[i]*self.H2c)*self.dt), output) #TODO IMPORTANT NEED TO CHANGE
        return output # may change

        

def fidelity(target, current):
    return -(abs(torch.trace(torch.matmul(torch.adjoint(current),target))/8))**2

def train(model, optim, target):
    # training data

    for i in range(200000):
        predictedOut = model()
        loss = fidelity(predictedOut, target).requires_grad_()

        if i % 100 == 99:
            print(f"Loss: {loss}")
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()




t = 1*10**(-8)
N = 5
dt = t/N
X = torch.tensor([[0, 1], [1, 0]])
Y = torch.tensor([[0, complex(0, 1)], [complex(0, 1), 0]])
Z = torch.tensor([[1, 0], [0, -1]])
I = torch.eye(2)
Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
H1c = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)
H0real = torch.eye(8)
H0real[6][6], H0real[6][7] = H0real[6][7], H0real[6][6]
H0real[7][6], H0real[7][7] = H0real[7][7], H0real[7][6]
H0 = torch.complex(H0real, torch.zeros(8))

model = QOC(Hd, H1c, H2c, dt, N)

adam = torch.optim.Adam(model.parameters())
train(model, adam, H0)



