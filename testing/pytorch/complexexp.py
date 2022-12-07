import torch
import math

z = torch.tensor([complex(0,math.pi), complex(1,1)])

print(torch.exp(z))