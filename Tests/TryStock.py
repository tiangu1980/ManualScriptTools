import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import argparse

# 读取股票数据
file_path = "600233.xlsx"
df = pd.read_excel(file_path, index_col="Date")

# 选择用于训练的特征（Close 和 Volume）
train_data = df[['Close', 'Volume']].values.astype(float)

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train_data)

# 创建窗口序列的函数
def create_sequences(data, window_size):
    sequences = []
    target = []
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        label = data[i + window_size]
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

# 构建 LSTM 模型
def build_lstm_model(input_shape, forget_length):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, recurrent_dropout=1.0 - forget_length))
    model.add(LSTM(units=50, recurrent_dropout=1.0 - forget_length))
    model.add(Dense(units=2))  # 输出 Close 和 Volume 两个特征
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 解析命令行参数
parser = argparse.ArgumentParser(description='股票预测模型训练和预测')
parser.add_argument('--forget_length', type=float, default=0.2, help='LSTM 遗忘长度')
parser.add_argument('--prediction_days', type=int, default=10, help='期待预测的天数')
args = parser.parse_args()

# 创建窗口序列
window_size = 60
X, y = create_sequences(train_data_normalized, window_size)

# 为 LSTM 重塑输入数据
X = np.reshape(X, (X.shape[0], X.shape[1], 2))

# 建立和训练模型
input_shape = (X.shape[1], 2)
model = build_lstm_model(input_shape, args.forget_length)
model.fit(X, y, epochs=200, batch_size=32)

# 预测未来几天的股价和交易量
last_window = train_data_normalized[-window_size:]
predictions = []

for _ in range(args.prediction_days):
    pred = model.predict(np.reshape(last_window, (1, window_size, 2)))
    predictions.append(pred[0])
    last_window = np.append(last_window[1:], pred, axis=0)

# 反向转换预测值为原始比例
predictions = scaler.inverse_transform(predictions)

# 打印每日预测值
for i, prediction in enumerate(predictions, start=1):
    print(f"Day {i} - Predicted Close: {prediction[0]}, Predicted Volume: {prediction[1]}")
