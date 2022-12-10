# I'll do some changes and prob make some more helper functions to make it better to maintain. (Kasper)

import torch

# Defined constants for changing learning rate
t = 1*10**(-8)
N = 5
dt = t/N
hbar = 1

# Define gates for quantum gates
X = torch.tensor([[0, 1], [1, 0]])
Y = torch.tensor([[0, complex(0, 1)], [complex(0, 1), 0]])
Z = torch.tensor([[1, 0], [0, -1]])
I = torch.eye(2)

# Target H-nought denoted as Hd
Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
H1c = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)

# initial conditions
real = torch.eye(8)
imag = torch.zeros(8)
U0 = torch.complex(real, imag)

# 
def U(n):
  return torch.linalg.matrix_exp(complex(0, -1)/hbar*(Hd + a1[n]*H1c + a2[n]*H2c)*dt)


def Uf():
  real = torch.eye(8) # don't know if these are necessary, as defined in initial conditions
  imag = torch.zeros(8)
  Ueq = torch.complex(real, imag)
  for i in range(N):
    Ueq = torch.matmul(U(i), Ueq)
  return Ueq


# main process
a1 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True)
a2 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True)
L = (abs(torch.trace(torch.matmul(torch.adjoint(Uf()),U0)))/8)**2
while L.item() > 0.1:
  L = (abs(torch.trace(torch.matmul(torch.adjoint(Uf()),U0)))/8)**2
  L.backward(retain_graph = True)
  print(a1.grad)
  a1 = a1-0.01*a1.grad #is the grad vector normalized, it actually shouldn't be normalized (from my understanding on 3b1b AI video on gradient)
  a2 = a2-0.01*a2.grad #a larger magnitude denotes a larger step so not too sure about why it needs to be normalized
  a1 = a1.clone().detach().requires_grad_(True)
  a2 = a2.clone().detach().requires_grad_(True)


  
