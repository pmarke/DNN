import torch 
import numpy as np 

## Initializing a tensor
data = [[1,2],[3,4]]
np_array = np.array(data)

x_data = torch.tensor(data)   # can initialize from a list
x_np = torch.tensor(np_array) # can initialize from a numpy array 

x_ones = torch.ones_like(x_data)                    # has the same shame and data type as x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the data type of x_data but keeps the shape

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

## Tensor attributes: shape, datatype, and the device on which they are stored
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

## Operations on tensors
# By default, tensors are created on teh CPU. We need to explicitly move tensors to GPU
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# in place operations are denoted by a _ suffix
print(tensor, "\n")
tensor.add_(5)
print(tensor)