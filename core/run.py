"""
For NMR 3-qubit use
With Saving
"""
import warnings
import torch
from pathlib import Path

from core import train, QOC

# ignore complex warnings
warnings.filterwarnings("ignore")

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

# Main Process
model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
adam = torch.optim.Adam(model.parameters(), lr = 0.001)

# high smoothness -> high weight
train(model=model, optim=adam, target=H0, accuracy=0.01, smoothness=0.875, maxiterations=100000, weight=0.4)
# save model
torch.save(model.state_dict(), str(Path(__file__).parent.absolute())+f"/results/state.pt")
# plot model
model.plot()