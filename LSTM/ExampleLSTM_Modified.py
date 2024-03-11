import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

df = pd.read_excel('600233.xlsx')
df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
timeseries = df[["Close"]].values.astype('float32')

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

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

lookback = 10
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 用从倒数第十一个开始的十个数据预测验证最后一个
#mystart=len(test) - lookback - 1
# 用与X_test相同的起始数据验证后续所有结果
mystart=len(test) - lookback - 1 - 84
print(f"mystart {mystart}")
X_test_me = create_dataset_single(test, lookback, idstart=mystart)

print(f"X_test    value {X_test}")
print(f"X_test_me value {X_test_me}")
#for i, value in enumerate(X_test):
#    print(f"X_test {i} 的值: {value}")

#print(f"X_test_me的 形状是： {X_test_me.shape} , 类型是： {type(X_test_me)} , 值是： {X_test_me}")
#for i, value in enumerate(X_test_me):
#    print(f"X_test_me {i}  形状是： {value.shape} ，类型是{type(value)} , 值是: {value} ")
    
#exit()

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]

print(f"test_plot 类型： {type(test_plot)}")
#print(f"test_plot   值： {test_plot}")

# 循环遍历 test_plot 数组中的每个元素，并打印出它们的值
for i, value in enumerate(test_plot):
    #print(f"test_plot {i} 的值: {value}")
    print(f"test_plot {i} 的值: {value[0]}")

#print(type(test_plot))
#print(test_plot.shape)

#my_Single_Result=model(X_test_me)
#print(type(my_Single_Result))
# my_Single_Result[:, -1, :][0][0] my_Single_Result最后一行数据，形状是(1,1)。加[0][0]得到它的纯值。
#print(my_Single_Result)
#print(my_Single_Result.shape)

myvals = []
my_plot = np.ones_like(timeseries) * np.nan
with torch.no_grad():
    for i in range(1, 286, 1):
        print(i)
        my_Single_Result=model(X_test_me)
        my_single_value=my_Single_Result[:, -1, :][0][0]
        #print(f"Pre {i} : {my_single_value}")
        myvals.append(my_single_value)
        # 删除第一个时间步的值
        X_test_me = X_test_me[:, 1:, :]
    
        # 添加预测值 my_single_value 到尾部
        new_value = torch.tensor([[[my_single_value]]])
        X_test_me = torch.cat((X_test_me, new_value), dim=1)


for i in range(201, 286, 1):
    my_plot[i] = myvals[i-201]

for i, value in enumerate(my_plot):
    #print(f"test_plot {i} 的值: {value}")
    print(f"my_plot {i} 的值: {my_plot[i][0]}")


# plot
plt.plot(timeseries, color="b")
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.plot(my_plot, c='k')
plt.savefig('600233.png')
plt.show()