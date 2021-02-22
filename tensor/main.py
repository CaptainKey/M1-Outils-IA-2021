import torch 
import numpy as np

# a = torch.tensor([1,2])

# print(a)
# print(type(a))

# a_numpy = a.numpy()

# print(a_numpy)
# print(type(a_numpy))

# one = torch.ones(100)
# print(one)

# one_numpy = np.ones(100)
# print(one_numpy)

x = torch.tensor([10.],requires_grad=True)
a = torch.tensor([2.],requires_grad=True)
b = torch.tensor([6.],requires_grad=True)

"""
    y = ax+b
    dy/dx = a 
    dy/da = x 
    dy/db = 1
"""

y = a*x+b
y_truth = 200

error = y - y_truth

print('y',y)

y.backward()

print('a.grad',a.grad)
print('b.grad',b.grad)

"""
n = 0.001
n = 0.01

dE/dW = 0 => W_(t+1) = W_t

W_(t+1) = W_t - n*dE/dW 
"""