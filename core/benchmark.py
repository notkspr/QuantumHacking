"""
For benchmarking (with qutip)
"""
import warnings
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import math

from core import QOC, train

# ignore complex warnings
warnings.filterwarnings("ignore")

def benchmark(n, N):
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

    # Target unitary: Hadamard gate on n qubits
    H = torch.tensor([[1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]], dtype=torch.complex64)
    U0 = torch.tensor([1], dtype = torch.complex64)
    for i in range(n):
        U0 = torch.kron(U0, H)

    model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)
    return train(model = model, optim = optim, target = U0, accuracy = 0.01, roughness = 10, weight = 1/N, maxiterations = 50, benchbool = True)

def printresult(n, N, result):
    print(f"Number of qubits: {n}\nNumber of time slices: {N}")
    print(f"Average time per epoch: {result} seconds\n")

xlist = []
ylist_model = []

def func(n, N):
    result = benchmark(n, N)
    printresult(n, N, result)
    xlist.append(n)
    ylist_model.append(result)

func(2, 20)
func(3, 80)
func(4, 300)
func(5, 1200)
func(6, 5000)

try:
    open(str(Path(__file__).parent.absolute())+f"/results/time/time.png", "x").close()
except:
    pass

plt.plot(xlist, ylist_model, color='r', label='Our model')
ylist_qutip = [0.0526, 0.2166, 0.9456, 5.8112, 92.0416] # values from qutip_benchmark.py
plt.plot(xlist, ylist_qutip, color='g', label='Qutip')
plt.yscale("log")
plt.title(f"Iteration time")
plt.xlabel("number of qubits N")
plt.ylabel("runtime per iteration (s)")
plt.xticks(xlist)
leg = plt.legend(loc='lower right')
plt.savefig(str(Path(__file__).parent.absolute())+f"/results/time/time.png")
plt.clf()
