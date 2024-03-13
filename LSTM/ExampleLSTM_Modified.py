import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import time

# 1 new data relayed on 10 previous data
lookback = 10

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

# Get last lookback-1 data from datasets to be a input, to predict the NO.lookback value
def create_dataset_single(dataset, count, idstart):
    X = []
    feature = dataset[idstart:idstart+count]
    X.append(feature)
    return torch.tensor(X)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def train_model(timeseries, model_file, n_epochs=10):
    if len(timeseries) < lookback + 1:
        print("train_model : Time series is too short for the lookback period.")
        return
    
    X_train, y_train = create_dataset(timeseries, lookback=lookback)
    
    model = AirModel()
    
    # 初始化或加载模型
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    execution_time = 0
    expect_time = 0
    cost_time = 0
    top_start_time = time.time()
    for epoch in range(n_epochs):        
        start_time = time.time()
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
        end_time = time.time()
        execution_time = end_time - start_time
        cost_time = end_time - top_start_time
        expect_time = cost_time/(epoch+1)*n_epochs
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item()} , Execution time: {execution_time:.2f} s, Cost time: {cost_time:.2f} s, Expect time: {expect_time:.2f} s")
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        
    torch.save(model.state_dict(), model_file)
    print("Model trained and saved successfully.")

def predict_model(timeseries, model_file, num_predictions):    
    if len(timeseries) < lookback:
        print("predict_model : Time series is too short for the lookback period.")
        return
    
    model = AirModel()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    mystart=len(timeseries)-lookback
    X_test_me = create_dataset_single(timeseries, lookback, idstart=mystart)
    
    print(f"Data start: {mystart}")
    #print(f"timeseries: {timeseries[mystart:]}")
    print(f"X_test_me: {X_test_me}")
    
    myvals = []
    pred_length = num_predictions
    with torch.no_grad():
        for i in range(0, pred_length, 1):
            my_Single_Result=model(X_test_me)
            my_single_value=my_Single_Result[:, -1, :][0][0]
            myvals.append(my_single_value)
            print(f"Pre {i} : {my_single_value}")
            
            # 删除第一个时间步的值
            X_test_me = X_test_me[:, 1:, :]
    
            # 添加预测值 my_single_value 到尾部
            new_value = torch.tensor([[[my_single_value]]])
            X_test_me = torch.cat((X_test_me, new_value), dim=1)
    
    return

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "--mode",      help="Mode: train or work",                       required=True)
    parser.add_argument("--f", "--file",      help="Data file path",                            required=True)
    parser.add_argument("--c", "--column",    help="Column to train and predict",               required=True)
    parser.add_argument("--h", "--head",      help="Number of head rows to keep in dataframe",  type=int, required=True)
    parser.add_argument("--e", "--end",       help="Number of end rows to keep in dataframe",   type=int, required=True)
    parser.add_argument("--t", "--times",     help="Number run times of training, or predicts", type=int, default=1)
    parser.add_argument("--n", "--namemodel", help="Model file name",                           required=True)
    args = parser.parse_args()

    # 读取数据文件
    df = pd.read_excel(args.f)
       
    # 保留指定行数数据
    if args.h < len(df) and args.h > 0:
        # 保留最开始 h 行数据
        df = df.head(args.h)
    
    # 保留指定行数数据
    if args.e < len(df) and args.e > 0:
        # 保留最后 e 行数据
        df = df.tail(args.e)

    # 保留指定列数据
    df = df.loc[:, [args.c]]  
    timeseries = df[[args.c]].values.astype('float32')

    print(f"Data len: {len(timeseries)}")
    #print(f"Data: {timeseries}")

    if args.m == "train":
        # 检查是否存在模型文件
        if args.n:
            train_model(timeseries, args.n, args.t)
        else:
            train_model(timeseries, None, args.t)
    elif args.m == "work":
        try:
            # 检查是否存在模型文件
            if args.n:
                predict_model(timeseries, args.n, args.t)
            else:
                print("Model file not found. Please train the model first.")
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
            return

# python ExampleLSTM_Modified.py --m train --f 600233.xlsx --c Close --h 9999 --e 9999 --t 1 --n 600233_1K_Close.amod
# python ExampleLSTM_Modified.py --m work  --f 600233.xlsx --c Close --h 9999 --e 9999 --t 5 --n 600233_1K_Close.amod

if __name__ == "__main__":
    main()
