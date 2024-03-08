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

    # 准备数据
    print("train_model df.shape[1]:", df.shape[1])
    data = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)

    # 初始化模型
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

    # 获取最新一行数据
    input_data = torch.tensor(df.iloc[-1, 1:].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)

    # 进行预测
    predictions = []

    predictions = []
    with torch.no_grad():
        for _ in range(num_predictions):
            output = model(input_data)
            predictions.append(output.numpy())
            input_data = torch.cat((input_data[:, 1:], output.unsqueeze(0)), dim=1)


    # 输出预测结果
    for pred in predictions:
        print(pred)

# 定义一个函数，将输入转换为float32  
def to_float32(value):  
    if isinstance(value, (int, float)): 
        return np.float32(value)  
    else:  
        # 如果不是数字，则返回原始值或根据需要处理  
        return value  

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Mode: train or work", required=True)
    parser.add_argument("--f", help="Data file path", required=True)
    parser.add_argument("--l", help="Number of rows to keep in dataframe", type=int, required=True)
    parser.add_argument("--p", help="Number of predictions to make", type=int, default=1)
    parser.add_argument("--n", help="Model file name", required=True)
    args = parser.parse_args()

    # 读取数据文件
    df = pd.read_excel(args.f)
    
    # 保留指定行数数据
    df = df.tail(args.l)
    
    df = df.applymap(to_float32)
    print("Main df.shape[1]:", df.shape[1])


    if args.m == "train":
        train_model(df, args.n)
    elif args.m == "work":
        try:
            predict_model(df, args.n, args.p)
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
            return

if __name__ == "__main__":
    main()



# 我有一套遵循日期顺序从最老到最新排列的数据，保存在xlsx文件里。
# 除了第一行是title外，后面的每一行都是一组当日的数据。第1列是日期类型的索引列。第2到5列是float类型。第6到7列是int类型。
# 每一行第2到7列数据的值，都由它前面的所有数据决定。
# 每一行第2到7列的数据之间都强相关。
# 
# 请给我一个这样的脚本，使用pytorch的LSTM来实现：
# 1. 根据调用脚本的参数“--m”来决定脚本是工作在"train"模式还是"work"模式。
# 2. 根据调用脚本的参数“--f”来读取指定数据文件到dataframe。
# 3. 根据调用脚本的参数“--l”来指定dataframe保留指定行数的数据，之后的数据全部从dataframe中删掉。
# 4. "train"模式时，使用dataframe中的所有数据作为训练数据进行训练，不必划分出测试集。
# 5. "work"模式时，使用dataframe中的最新一行第2到7列数据作为模型的一组输入数据，得到预测时间序列上下一天的预测数据。如果脚本的参数“--p”的值大于1，则将上一次的预测数据作为输入来获得再下一天的预测数据。直到获得“--p”指定数量的预测数据。得到的预测数据必须也有6列数据，这6列数据与输入数据的列顺序关系相同。
# 5. 根据调用脚本的参数“--n”来指定模型文件的名字。在启动脚本时，读取这个模型文件。"train"模式时，如果不存在这个文件，则新建模型文件进行训练。在训练结束之后，将模型在当前目录保存为这个名字。"work"模式时，如果不存在这个文件，则打印信息并直接退出脚本；如果模型文件存在，则使用这个模型根据第5条要求进行预测。