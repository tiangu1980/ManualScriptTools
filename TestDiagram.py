import torch

# 创建两个Tensors
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 定义计算图
z = x * y

# 进行更多操作
w = z + x

# 计算梯度
w.backward()

# 输出梯度
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
