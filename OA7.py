import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入必要的库

# 2. 从 Excel 文件中加载数据
data = pd.read_excel("data.xlsx", header=0)
dates = data.iloc[1:92, 0].values
y_true = data.iloc[1:92, 1].values

# 3. 准备训练数据
train_data = data.iloc[2:62, 1].values
test_data = data.iloc[62:92, 1].values
train_data = torch.FloatTensor(train_data).view(-1)
test_data = torch.FloatTensor(test_data).view(-1)

# 4. 创建时间序列数据
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_window = 10
train_inout_seq = create_inout_sequences(train_data, train_window)

# 5. 创建简单的 LSTM 模型
class LSTM(nn.Module):
    # ...

model = LSTM(input_dim=1, hidden_dim=64, output_dim=1, num_layers=1)

# 6. 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(100):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()

# 7. 使用训练好的模型进行预测
model.eval()
test_inout_seq = create_inout_sequences(test_data, train_window)
y_pred = []

for seq, _ in test_inout_seq:
    seq = seq.view(-1)
    with torch.no_grad():
        y_pred.append(model(seq).item())

# 8. 计算偏差方差
mse = ((y_pred - y_true) ** 2).mean()

# 9. 绘制图表
plt.figure(figsize=(16, 10))
plt.plot(dates, y_true, label="True Data", color="blue")
plt.plot(dates[10:], y_pred, label="Predicted Data", color="red")
plt.title(f"Prediction vs True Data - MSE: {mse:.6f}")
plt.xlabel("Datetime")
plt.ylabel("Value")
plt.legend()
plt.show()
