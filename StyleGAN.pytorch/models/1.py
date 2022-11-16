import torch 
import numpy as np

kernel = torch.tensor([1, 2, 1], dtype=torch.float32)

print(kernel[:, None])

kernel = kernel[:, None] * kernel[None, :]
print(kernel)
print(kernel[None, None].shape)

x = torch.randn(1, 3, 2)
print(x[:, 0])
print(x)
print(np.sqrt(2))