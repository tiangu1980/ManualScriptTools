import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 读取数据
def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 数据预处理
def preprocess_data(data):
    # 将'date'列转换为datetime类型
    data['date'] = pd.to_datetime(data['date'])
    
    # 使用日期列作为索引
    data.set_index('date', inplace=True)
    
    # 归一化数据，将'level'列的值缩放到[0, 1]范围
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['level'] = scaler.fit_transform(data[['level']])
    
    return data, scaler

# 构建训练数据
def build_training_data(data, time_steps=4):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:(i + time_steps)].values)
        y.append(data.iloc[i + time_steps]['level'])
    X, y = np.array(X), np.array(y)
    print(f"X.shape {X.shape}, y.shape {y.shape}")
    return X, y

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 训练模型
def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# 预测未来数据
def predict_future(model, data, scaler, time_steps=4, future_steps=5):
    x_input = data.iloc[-time_steps:].values
    x_input = x_input.reshape((1, time_steps, 1))
    
    future_predictions = []
    for i in range(future_steps):
        prediction = model.predict(x_input, verbose=0)
        future_predictions.append(prediction[0])
        
#        x_input = np.append(x_input[0, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        # 调整为匹配维度
        if len(x_input.shape) == 3 and len(prediction.shape) == 2:
            x_input = np.concatenate((x_input[0, 1:], prediction.reshape(1, 1)), axis=0)
        elif len(x_input.shape) == 3 and len(prediction.shape) == 3:
            x_input = np.concatenate((x_input[0, 1:], prediction.reshape(1, 1, 1)), axis=0)
        elif len(x_input.shape) == 2 and len(prediction.shape) == 2:
            x_input = np.concatenate((x_input[1:], prediction.reshape(1, 1)), axis=0)
        elif len(x_input.shape) == 2 and len(prediction.shape) == 3:
            x_input = np.concatenate((x_input, prediction.reshape(1, 1, 1)), axis=0)
        x_input = x_input.reshape((1, time_steps, 1))
    
    # 反向转换预测结果
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions




# 主程序
if __name__ == '__main__':
    # 文件路径
    file_paths = ['fake_data_1.csv', 'fake_data_2.csv', 'fake_data_3.csv', 'fake_data_4.csv', 'fake_data_5.csv']
    
    # 读取和预处理数据
    datasets = [preprocess_data(read_data(file_path)) for file_path in file_paths]
    print(len(datasets))
    
    # 构建训练数据
    time_steps = 4
    X_train, y_train = zip(*[build_training_data(data, time_steps) for data, _ in datasets])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # 构建和训练LSTM模型
    model = build_lstm_model(input_shape=(time_steps, 1))
    train_model(model, X_train, y_train)
    
    # 预测未来数据
    future_predictions = [predict_future(model, data, scaler, time_steps) for data, scaler in datasets]
    
    # 打印未来预测值
    for i, (_, scaler) in enumerate(datasets):
        print(f"\n预测结果（数据集{i + 1}）:")
        print(scaler.inverse_transform(future_predictions[i]))
