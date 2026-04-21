import torch
import numpy as np
import time


N = 1000000 

list_one = list(range(N))
list_two = list(range(N))

start_time = time.time()
python_list_result = [a + b for a, b in zip(list_one, list_two)]
end_time = time.time()
list_time = end_time - start_time
print(f"Python List took: {list_time:.6f} seconds")

numpy_array_one = np.arange(N)
numpy_array_two = np.arange(N)

start_time = time.time()
numpy_array_result = numpy_array_one + numpy_array_two
end_time = time.time()
numpy_time = end_time - start_time
print(f"Numpy Array took: {numpy_time:.6f} seconds")

pytorch_tensor_one = torch.arange(N)
pytorch_tensor_two = torch.arange(N)

start_time = time.time()
pytorch_tensor_result = pytorch_tensor_one + pytorch_tensor_two
end_time = time.time()
tensor_time = end_time - start_time
print(f"Pytorch Tensor (CPU) took: {tensor_time:.6f} seconds")


tensor_1d_torch = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {tensor_1d_torch}")

tensor_2d_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"2D Tensor:\n{tensor_2d_torch}")

tensor_3d_torch = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor:\n{tensor_3d_torch}")

tensor_nd_torch = torch.randn(2, 3, 4, 5)
print(f"nD Tensor shape: {tensor_nd_torch.shape}")


tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([4, 5, 6])
tensor_sum = tensor_a + tensor_b

tensor_diff = tensor_a - tensor_b

tensor_mul = tensor_a * tensor_b

tensor_div = tensor_a / tensor_b

tensor_dot = torch.dot(tensor_a, tensor_b)

tensor_matmul = torch.matmul(tensor_2d_torch, tensor_2d_torch.T)

print(f"Tensor Sum: {tensor_sum}")
print(f"Tensor Difference: {tensor_diff}")
print(f"Tensor Multiplication: {tensor_mul}")
print(f"Tensor Division: {tensor_div}")
print(f"Tensor Dot Product: {tensor_dot}")
print(f"Tensor Matrix Multiplication:\n{tensor_matmul}")


mask = tensor_1d_torch > 2
masked_tensor = tensor_1d_torch[mask]

sub_tensor = tensor_2d_torch[:, 1:]

concatenated_tensor = torch.cat((tensor_a, tensor_b), dim=0)

stacked_tensor = torch.stack((tensor_a, tensor_b), dim=0)

print(f"Masked Tensor: {masked_tensor}")
print(f"Sub Tensor:\n{sub_tensor}")
print(f"Concatenated Tensor: {concatenated_tensor}")
print(f"Stacked Tensor:\n{stacked_tensor}")


viewed_tensor = tensor_2d_torch.view(3, 2)

reshaped_tensor = tensor_2d_torch.reshape(3, 2)

unsqueezed_tensor = tensor_1d_torch.unsqueeze(0)

squeezed_tensor = unsqueezed_tensor.squeeze(0)

print(f"Viewed Tensor:\n{viewed_tensor}")
print(f"Reshaped Tensor:\n{reshaped_tensor}")
print(f"Unsqueezed Tensor:\n{unsqueezed_tensor}")
print(f"Squeezed Tensor:\n{squeezed_tensor}")


tensor_c = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_tensor = tensor_c.reshape(3, 2)

print(f"Original Tensor:\n{tensor_c}")
print(f"Reshaped Tensor:\n{reshaped_tensor}")

numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
numpy_reshaped = numpy_array.reshape(3, 2)

print(f"Original Numpy Array:\n{numpy_array}")
print(f"Numpy Reshaped Array:\n{numpy_reshaped}")


tensor_c = torch.tensor([[1], [2], [3]])
tensor_d = torch.tensor([10, 20, 30])
broadcasted_sum = tensor_c + tensor_d

print(f"Broadcasted Sum:\n{broadcasted_sum}")


tensor_inplace = torch.tensor([1, 2, 3])
tensor_inplace.add_(5)

print(f"In-place Modified Tensor: {tensor_inplace}")

tensor_outofplace = torch.tensor([1, 2, 3])
tensor_new = tensor_outofplace + 5

print(f"Out-of-place New Tensor: {tensor_new}")
