import tensorflow as tf
import numpy as np
import argparse
import os
import pandas as pd

# 定义模型
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=5)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(3, activation='linear')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.flatten_layer(x)
        return self.dense_layer(x)

# 解析命令行参数
parser = argparse.ArgumentParser(description='TensorFlow 训练和工作模式脚本')
parser.add_argument('mode', choices=['train', 'work'], help='选择模式：train 或 work')
parser.add_argument('input_file', type=str, help='输入数据文件路径')
parser.add_argument('--model_path', type=str, default='trained_model', help='训练模式下保存/加载模型的路径')
args = parser.parse_args()

# 创建模型
model = MyModel()

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# 定义训练步骤
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 训练模式
if args.mode == 'train':
    # 读取训练数据
    try:
        df = pd.read_csv(args.input_file)
        # 使用简单的哈希函数确保结果在 [0, 10000) 范围内
        inputs = df.iloc[:, :5].apply(lambda col: col.map(lambda x: hash(str(x)) % 10000)).values
        targets = df.iloc[:, 5:].values
    except Exception as e:
        print("训练数据文件格式错误。请确保每行有五个以制表符分隔开的数据为输入数据，之后是用制表符分隔开的三个整型数。")
        exit()

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    for epoch in range(100):  # 100 次迭代，可根据实际情况调整
        for i in range(len(inputs)):
            input_seq = np.array(inputs[i], dtype=int)
            target_seq = np.array(targets[i], dtype=int)
            loss = train_step(input_seq.reshape((1, 5)), target_seq.reshape((1, 3)))

    # 保存模型
    model.save(args.model_path)

# 工作模式
elif args.mode == 'work':
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print("错误：找不到已训练好的模型。请先运行脚本以训练模型。")
        exit()

    # 加载模型
    model = tf.keras.models.load_model(args.model_path)

    # 读取工作数据
    try:
        df = pd.read_csv(args.input_file)
        # 使用相同的哈希函数处理工作数据
        inputs = df.iloc[:, :5].apply(lambda col: col.map(lambda x: hash(str(x)) % 10000)).values
    except Exception as e:
        print("工作数据文件格式错误。请确保每行有五个以制表符分隔开的数据为输入数据。")
        exit()

    # 预测并打印结果
    for i in range(len(inputs)):
        # 模拟一个5维的输入数据
        input_seq = np.zeros(5, dtype=int)
        input_seq[:3] = inputs[i]  # 将前3维赋值为实际的输入数据
        prediction = model.predict(input_seq.reshape((1, 5)))
        print(f"{df.iloc[i, 0]}\t{df.iloc[i, 1]}\t{df.iloc[i, 2]}\t{int(prediction[0, 0])}\t{int(prediction[0, 1])}\t{int(prediction[0, 2])}")





# 请给我一个使用tensorflow进行训练的python脚本，可以在调用它时使用参数选择“训练模式”或者“工作模式”。
# 在训练模式下读取一个文本文件，每行有三条字符串为一组输入数据和5个在有限范围内的输出整型数，以"\t"符号分开；训练目的是可以根据输入的两条字符串预测到它的5个输出整型数字，在训练结束后保存训练好的模型，下次重新训练时读入模型并继续训练。
# 在工作模式下读取一个文本文件，每行有三条字符串为一组输入数据，要求每一行都使用训练好的模型得到5个输出整型数，然后将输入数据和输出数据以"\t"符号分隔按行打印在屏幕上。
# 输入的文本文件不符合当前模式需求时，直接打印提示信息并退出脚本。
#
# python Csv3StrsIn5IntOutTensorflow.py train input_train.csv --model_path trained_model
# python Csv3StrsIn5IntOutTensorflow.py work input_work.csv --model_path trained_model