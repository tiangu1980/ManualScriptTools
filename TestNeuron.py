import numpy as np

class Neuron:
    def __init__(self, input_size):
        # 随机初始化权重和偏置
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def activate(self, inputs):
        # 计算加权和
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # 使用激活函数（例如sigmoid）处理加权和
        activation = self.sigmoid(weighted_sum)
        return activation

    def sigmoid(self, x):
        # Sigmoid激活函数
        return 1 / (1 + np.exp(-x))

# 创建一个具有3个输入的神经元
input_size = 3
neuron = Neuron(input_size)

# 生成一个包含3个随机输入的示例
inputs = np.random.rand(input_size)

# 计算神经元的输出
output = neuron.activate(inputs)

# 打印结果
print("Inputs:", inputs)
print("Weights:", neuron.weights)
print("Bias:", neuron.bias)
print("Output:", output)
