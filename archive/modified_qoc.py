# Original code of "qoc.py" does not work properly (gives nan+nanj --> not a number)
# This code replaces some parts of "qoc.py" with PyTorch algorithms
# This code has error "RuntimeError: expected scalar type ComplexFloat but found Float"
# We also need to change the optimizer to L-BFGS-B Optimizer, sample: https://gist.github.com/tuelwer/0b52817e9b6251d940fd8e2921ec5e20

import torch
import torch.nn as nn

# Define constants
t = 1*10**(-8)
N = 5
dt = t/N
hbar = 1.05*10**(-34)

# Define gates for quantum gates
X = torch.tensor([[0, 1], [1, 0]])
Y = torch.tensor([[0, complex(0, 1)], [complex(0, 1), 0]])
Z = torch.tensor([[1, 0], [0, -1]])
I = torch.eye(2)

# Define Hd, H1c, H2c
Hd = -1397*torch.kron(torch.kron(Z, I), I) + 29.95*torch.kron(torch.kron(I, Z), I) + 1015*torch.kron(torch.kron(I, I), Z) - 130.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, Z), I)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, X), I)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, Y), I))) + 50.2*(torch.matmul(torch.kron(torch.kron(Z, I), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(X, I), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(Y, I), I), torch.kron(torch.kron(I, I), Y))) + 68.45*(torch.matmul(torch.kron(torch.kron(I, Z), I), torch.kron(torch.kron(I, I), Z)) + torch.matmul(torch.kron(torch.kron(I, X), I), torch.kron(torch.kron(I, I), X)) + torch.matmul(torch.kron(torch.kron(I, Y), I), torch.kron(torch.kron(I, I), Y)))
H1c = torch.kron(torch.kron(X, I), I) + torch.kron(torch.kron(I, X), I) + torch.kron(torch.kron(I, I), X)
H2c = torch.kron(torch.kron(Y, I), I) + torch.kron(torch.kron(I, Y), I) + torch.kron(torch.kron(I, I), Y)

# Initial conditions
real = torch.eye(8)
imag = torch.zeros(8)
U0 = torch.complex(real, imag)
a1 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True)
a2 = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True)

# Define U(n) and Uf()
def U(n):
  return torch.exp(complex(0, -1)/hbar*(Hd + a1[n]*H1c + a2[n]*H2c)*dt)


def Uf():
  Ueq = torch.complex(real, imag)
  for i in range(N):
    Ueq = torch.matmul(U(i), Ueq)
  return Ueq

# 0) Training samples
# here, it is quite strange as U0 seems to be fixed in qoc.py? 
n_samples, n_features = U0.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) Create a test sample
real_test = torch.zeros(8)
imag_test = torch.zeros(8)
U0_test = torch.complex(real_test, imag_test)

# 1) Design model
input_size = n_features
output_size = n_features

# 1) Call the model with samples U0
model = nn.Linear(input_size, output_size)
print(f'Prediction before training: f(U0_test) = {model(U0_test).item():.3f}')


# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass
    U_predicted = model(U0)

    # loss
    l = (torch.log10(abs(torch.trace(torch.matmul(torch.adjoint(Uf()),U0)))/8))**2

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [a1, a2] = model.parameters() # unpack parameters
        print('epoch', epoch+1, ': a1 = ', a1[0][0].item(), 'loss = ', 1)

print(f'Prediction after training: f(U0) = {model(U0_test).item():.3f}')




