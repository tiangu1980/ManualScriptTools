import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 生成日期范围
date_range = pd.date_range(start='2022-01-01', periods=73)

# 生成伪数据
def generate_fake_data(date_range):
    # 生成倍率为1.1的等比数列
    ratio = 1.1
    base_value = 100  # 基础值
    values = base_value * np.power(ratio, np.arange(len(date_range)))
    
    # 生成正弦曲线和余弦曲线作为均线
    sine_wave = np.sin(np.linspace(0, 2 * np.pi, len(date_range)))
    cosine_wave = np.cos(np.linspace(0, 2 * np.pi, len(date_range)))
    
    # 正态分布的噪声
    noise = np.random.normal(loc=0, scale=5, size=len(date_range))
    
    # 生成5组数据
    datasets = []
    for i in range(5):
        # 使用正弦曲线或余弦曲线作为均线
        mean_line = sine_wave if i % 2 == 0 else cosine_wave
        
        # 生成带有噪声的数据
        data = pd.DataFrame({
            'date': date_range,
            'level': values * (1 + 0.1 * i) + noise + mean_line
        })
        
        datasets.append(data)
    
    return datasets

# 绘制伪数据的图表
def plot_fake_data(datasets):
    plt.figure(figsize=(12, 6))
    for i, data in enumerate(datasets):
        plt.plot(data['date'], data['level'], label=f'Dataset {i+1}')

    plt.title('Generated Fake Data')
    plt.xlabel('Date')
    plt.ylabel('Level')
    plt.legend()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 生成伪数据
    fake_datasets = generate_fake_data(date_range)
    
    # 绘制伪数据图表
    plot_fake_data(fake_datasets)

    # 保存伪数据到CSV文件
    for i, data in enumerate(fake_datasets):
        data.to_csv(f'fake_data_{i + 1}.csv', index=False)
