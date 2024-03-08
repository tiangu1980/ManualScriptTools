import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        output = self.fc(lstm_out.view(len(input), -1))
        return output[-1]

# 训练模型
def train_model(df, model_file):
    # 超参数
    input_size = 6
    hidden_size = 64
    output_size = 6
    learning_rate = 0.001
    num_epochs = 10 #100

    # 数据处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, 1:].values)

    # 转换为Tensor
    data = torch.tensor(scaled_data, dtype=torch.float32)

    # 初始化或加载模型
    if model_file:
        model = LSTMModel(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(model_file))
    else:
        model = LSTMModel(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data[:-1])
        loss = criterion(outputs, data[1:])
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # 保存模型
    torch.save(model.state_dict(), model_file)
    print("Model trained and saved successfully.")

# 预测模型
def predict_model(df, model_file, num_predictions):
    # 超参数
    input_size = 6
    hidden_size = 64
    output_size = 6

    # 初始化模型
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # 数据处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, 1:].values)

    # 获取最新一行数据
    input_data = torch.tensor(scaled_data[-1], dtype=torch.float32).unsqueeze(0)

    # 进行预测
    predictions = []

    with torch.no_grad():
        for _ in range(num_predictions):
            output = model(input_data)
            predictions.append(output.numpy())
            input_data = torch.cat((input_data[:, 1:], output.unsqueeze(0)), dim=1)

    # 还原预测数据结果
    predictions = scaler.inverse_transform(predictions)

    # 输出预测结果
    for pred in predictions:
        print(pred)

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Mode: train or work", required=True)
    parser.add_argument("--f", help="Data file path", required=True)
    parser.add_argument("--l", help="Number of rows to keep in dataframe", type=int, required=True)
    parser.add_argument("--p", help="Number of predictions to make", type=int, default=1)
    parser.add_argument("--n", help="Model file name", required=False)
    args = parser.parse_args()

    # 读取数据文件
    df = pd.read_excel(args.f)
    
    # 保留指定行数数据
    df = df.tail(args.l)

    if args.m == "train":
        # 检查是否存在模型文件
        if args.n:
            train_model(df, args.n)
        else:
            train_model(df, None)
    elif args.m == "work":
        try:
            predict_model(df, args.n, args.p)
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
            return

if __name__ == "__main__":
    main()
