#testing for nn.linear
import torch

# init complex tensor
targetreal = torch.eye(8)
targetcomplex = torch.zeros(8)
target = torch.complex(targetreal, targetcomplex)
_, targetfeatures = target.shape
print(f"Target dtype: {target.dtype}")

# testing tensor
testingreal = torch.zeros(8)
testingcomplex = torch.zeros(8)
testing = torch.complex(testingreal, testingcomplex)

# init nn.linear

model = torch.nn.Linear(targetfeatures, targetfeatures)

print(f"model dtype: {model.weight.dtype}")
print(f"Prediction before training: {model(testing)}")