import torch

# Total run time (quantum)
t = 1
# Number of slices of t
N = 10
# Time interval
dt = t/N

# Quantum gates
X = torch.tensor([[0., 1.], [1., 0.]])
Y = torch.tensor([[0., complex(0., -1.)], [complex(0., 1.), 0.]])
Z = torch.tensor([[1., 0.], [0., -1.]])
I = torch.eye(2)

# Drift Hamiltonian H_d
Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
# Control Hamiltonian 1 H^1_c
H1cr = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H1ci = torch.zeros(8)
H1c = torch.complex(H1cr, H1ci)
# Control Hamiltonian 2 H^2_c
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)

# Target unitary U0
U0r = torch.eye(8)
U0i = torch.zeros(8)
U0 = torch.complex(U0r, U0i)

# Product of unitaries, i.e. U_n*U_(n-1)*...*U(0)
def Uf():
  r = torch.eye(8)
  i = torch.zeros(8)
  Ueq = torch.complex(r, i)
  for i in range(N):
    Ueq = torch.matmul(torch.linalg.matrix_exp(complex(0, -1)*(Hd + a1[i]*H1c + a2[i]*H2c)*dt), Ueq)
  return Ueq


# Main process (tuning the amplitudes a1, a2)
# a1
a1r = torch.rand(N)
a1i = torch.zeros(N)
a1 = torch.complex(a1r, a1i).requires_grad_() # amplitude for control Hamiltonian H^1_c
# a2
a2r = torch.rand(N)
a2i = torch.zeros(N)
a2 = torch.complex(a2r, a2i).requires_grad_() # amplitude for control Hamiltonian H^2_c

L = abs(torch.trace(torch.matmul(torch.adjoint(Uf()), U0))/8-1)**2 # fidelity, min. (0) is achieved when product of unitaries = target matrix
r = 1 # learning rate
k = 0 # epoch

while L.item() > 10**(-10):
  L = abs(torch.trace(torch.matmul(torch.adjoint(Uf()), U0))/8-1)**2
  if k % 10 == 0:
    Litem = L.item()
    P_U = str(Uf())
    print(f"Epoch: {k} ")
    print(f"Loss: {Litem}")
    print("Product of unitaries:")
    print(f"{P_U}")
  L.backward(retain_graph = True)
  a1 = torch.sub(a1, a1.grad, alpha = r)
  a2 = torch.sub(a2, a2.grad, alpha = r)
  a1 = a1.clone().detach().requires_grad_(True)
  a2 = a2.clone().detach().requires_grad_(True)
  k += 1 

# Final result
print(f"Epoch: {k} ")
print(f"Loss: {L.item()}")
print("Fidelity: " + str(abs(torch.trace(torch.matmul(torch.adjoint(Uf()), U0)).item()/8)**2))
print("Product of unitaries:")
print(f"{Uf()}")
print("U gates:")
for i in range(N):
  print(f"U({i}): " + str(torch.linalg.matrix_exp(complex(0, -1)*(Hd + a1[i]*H1c + a2[i]*H2c)*dt)))
