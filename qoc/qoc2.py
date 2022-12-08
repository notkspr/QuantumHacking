# Added and changed some comments. (TCH)
# Corrected some mistakes.
# Autograd does not work.

import torch

# Total run time (quantum)
t = 1*10**(-5)
# Number of slices of t
N = 5
# Time interval
dt = t/N

# Define gates for quantum gates
X = torch.tensor([[0, 1], [1, 0]])
Y = torch.tensor([[0, complex(0, -1)], [complex(0, 1), 0]])
Z = torch.tensor([[1, 0], [0, -1]])
I = torch.eye(2)

# Drift Hamiltonian H_d
Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
# Control Hamiltonian H_c
H1c = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)

# Target unitary U0
real = torch.kron(torch.kron(X, I), X)
imag = torch.zeros(8)
U0 = torch.complex(real, imag)

# Unitary at time t_n (nth slice)
def U(n):
  return torch.linalg.matrix_exp(complex(0, -1)*(Hd + a1[n]*H1c + a2[n]*H2c)*dt)


# Product of unitaries, i.e. U_n*U_(n-1)*...*U(0)
def Uf():
  real = torch.eye(8)
  imag = torch.zeros(8)
  Ueq = torch.complex(real, imag)
  for i in range(N):
    Ueq = torch.matmul(U(i), Ueq)
  return Ueq


# main process (tuning the amplitude a1, a2)
a1 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True) # amplitude for control Hamiltonian H^1_c
a2 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True) # amplitude for control Hamiltonian H^2_c
L = (abs(torch.trace(torch.matmul(torch.adjoint(Uf()),U0)))/8-1)**2 # modified fidelity, min. (0) is achieved when product of unitaries = target matrix
r = 0.01 # learning rate
k = 0 # epoch

while L.item() > 0.01:
  if k % 10 == 0:
    print(f"Epoch: {k} " + str(Uf()))
  L = (abs(torch.trace(torch.matmul(torch.adjoint(Uf()),U0)))/8-1)**2
  print(L.item()) # show value of modified fidelity
  L.backward(retain_graph = True)
  a1 = a1-r*a1.grad
  a2 = a2-r*a2.grad
  a1 = a1.clone().detach().requires_grad_(True)
  a2 = a2.clone().detach().requires_grad_(True)
  k += 1
