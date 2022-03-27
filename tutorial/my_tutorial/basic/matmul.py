import torch

a = torch.arange(12).reshape(2,2,3)
b = torch.arange(12).reshape(2,3,2)

print(a)
print(b)

c = torch.matmul(a,b)
print(c.shape)