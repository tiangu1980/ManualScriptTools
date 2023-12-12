import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# 读取数据
df = pd.read_excel('data.xlsx')

# 将数据转换为numpy数组
data = df.iloc[:, 1:].values

# 划分训练集和测试集
train_data = torch.FloatTensor(data[1:61])
test_data = torch.FloatTensor(data[61:92])

# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_layer_size=100, output_size=6):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 训练模型
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150

for i in range(epochs):
    for seq in train_data:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, seq[-1].unsqueeze(0))  # 调整目标值的形状
        single_loss.backward()
        optimizer.step()

    if i%25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 预测
test_inputs = train_data[-60:].tolist()
model.eval()

for i in range(31):
    seq = torch.FloatTensor(test_inputs[-60:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = test_inputs[-31:]

# 绘制结果
x = np.arange(62, 93, 1)
plt.figure(figsize=(16.8, 10.5))
plt.title('Time Series Prediction')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df.iloc[:, 0][61:92], actual_predictions)
plt.plot(df.iloc[:, 0][61:92], test_data)
plt.show()
