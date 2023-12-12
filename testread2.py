import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
df = pd.read_excel('data.xlsx')
print(df.head(3))

# 选择第2到7列，从第二行开始
#df = df.iloc[1:, 1:7]
# 选择第2到7列，从第一行开始
df = df.drop(df.columns[0], axis=1)

# 将DataFrame转换为Tensor
#tensor = torch.tensor(df.values.astype(float))

print(df.head(3))

#print(df.columns[0])

print(df.iloc[0, 0])

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.layers(x)

# 创建模型和优化器
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 准备数据
train_data = torch.tensor(df.iloc[:150].values, dtype=torch.float32)
test_data = torch.tensor(df.iloc[150:187].values, dtype=torch.float32)

# 训练模型
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(train_data)
    loss = loss_fn(output, train_data)
    loss.backward()
    optimizer.step()

# 预测并计算方差
predictions = model(test_data)
variance = ((predictions - test_data) ** 2).mean().item()

print('方差: ', variance)

# 获取列名
column_name = df.columns[0]

rel_data=test_data[:, 1]
pre_data=predictions[:, 1]

print(rel_data[1])
print(pre_data[1])

# 绘制图形
plt.figure(figsize=(1080/80, 1024/80))
#plt.plot(range(30, 45), test_data.numpy(), label='真实值')
#plt.plot(range(30, 45), predictions.detach().numpy(), label='预测值')
#plt.plot(range(150,187), test_data.numpy(), label=f'真实值-{column_name}')
#plt.plot(range(150,187), predictions.detach().numpy(), label=f'预测值-{column_name}')
plt.plot(range(150,187), rel_data.detach().numpy(), label=f'真实值-{column_name}')
plt.plot(range(150,187), pre_data.detach().numpy(), label=f'预测值-{column_name}')
plt.legend()
plt.show()
