"""
For comparing with (with qutip)
"""
import warnings
import torch
from pathlib import Path

from core import QOC, trainwithtiming

# ignore complex warnings
warnings.filterwarnings("ignore")

n = 3
N = int(5*(4**n))
dt = 0.0001

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
# 3 qubits
F = torch.tensor([[-1397, -130.2, 50.2], [0, 29.95, 68.45], [0, 0, 1015]], dtype=torch.complex64)/2
# n qubits, normally F is a given nxn matrix, but for now all coefficients are set to 1/2; converges but fidelity does not approach 1
'''
F = torch.ones(n, n, dtype=torch.complex64)/2
for i in range(n):
    for j in range(n):
        if i > j:
            F[i][j] = 0
'''

H2d1 = lambda gate : torch.einsum("ij, ikl, jlm -> km", F, gate, gate)
H2d2 = lambda gate : torch.einsum("ii, ikl, ilm -> km", F, gate, gate)

Hd = torch.einsum("ii, ijk -> jk", F, Zs) + H2d1(Zs) + H2d1(Xs) + H2d1(Ys) - H2d2(Zs) - H2d2(Xs) - H2d2(Ys)
Hc = torch.stack([torch.einsum("ijk -> jk", Xs), torch.einsum("ijk -> jk", Ys)])
maxpower = 1500

# Target unitary
H0 = torch.eye(2**n, dtype=torch.complex64)
H0[6][6] = 0
H0[6][7] = 1
H0[7][7] = 0
H0[7][6] = 1


model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
trainwithtiming(model = model, optim = optim, target = H0, requiredaccuracy = 0.01, penaltyconst = 2, weight = 1/N, maxiterations = 100000)
