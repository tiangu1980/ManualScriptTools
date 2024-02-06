import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, ff_dim, output_dim):
        super(TransformerModel, self).__init__()

        self.embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.encoder_layers = [self.get_encoder_layer(d_model, num_heads, ff_dim) for _ in range(num_encoder_layers)]
        self.decoder_layers = [self.get_decoder_layer(d_model, num_heads, ff_dim) for _ in range(num_decoder_layers)]
        self.final_layer = layers.Dense(output_dim, activation='linear')

    def get_encoder_layer(self, d_model, num_heads, ff_dim):
        return tf.keras.Sequential([
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
            layers.Dropout(0.1),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(0.1),
            layers.LayerNormalization(epsilon=1e-6)
        ])

    def get_decoder_layer(self, d_model, num_heads, ff_dim):
        return tf.keras.Sequential([
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
            layers.Dropout(0.1),
            layers.LayerNormalization(epsilon=1e-6),
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
            layers.Dropout(0.1),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(0.1),
            layers.LayerNormalization(epsilon=1e-6)
        ])

    def call(self, inputs, training):
        x = self.embedding_layer(inputs)
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        for decoder in self.decoder_layers:
            x = decoder(x, training=training)
        return self.final_layer(x)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"错误：无法读取数据文件：{e}")
        exit()

def preprocess_data(data):
    # 处理字符串列中的空值
    string_columns = data.select_dtypes(include='object').columns
    data[string_columns] = data[string_columns].replace('', 'None')

    # 处理数字列中的空值
    numeric_columns = data.select_dtypes(include=['int', 'float']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

    return data

def train_model(model, train_data, val_data, epochs=10, model_path='transformer_model.h5'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    X_train, X_val, y_train, y_val = train_test_split(train_data.iloc[:, :3], train_data.iloc[:, 3:], test_size=0.2, random_state=42)

    # 处理数据中的空值
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)

    # 将浮点数转换为字符串
    X_train = X_train.astype(str)
    X_val = X_val.astype(str)

    # 在这里使用 TextVectorization 层将文本转换为整数序列
    text_vectorizer = layers.TextVectorization(max_tokens=1000, output_mode='int')
    text_vectorizer.adapt(X_train.values.flatten())

    model.fit(text_vectorizer(X_train), y_train, epochs=epochs, validation_data=(text_vectorizer(X_val), y_val))

    # 保存模型
    model.save(model_path)
    print(f"模型已保存到：{model_path}")


def work_mode(model, test_data):
    # 读取模型
    try:
        model = tf.keras.models.load_model('transformer_model.h5')
    except Exception as e:
        print(f"错误：找不到已训练好的模型：{e}")
        exit()

    # 处理测试数据中的空值
    test_data = preprocess_data(test_data)

    # 推理并打印结果
    text_vectorizer = layers.TextVectorization(max_tokens=1000, output_mode='int')
    text_vectorizer.adapt(test_data.values.flatten())

    for i in range(len(test_data)):
        input_data = np.array(text_vectorizer(test_data.iloc[i, :3])).reshape(1, -1)
        prediction = model.predict(input_data)
        print(f"{test_data.iloc[i, 0]}\t{test_data.iloc[i, 1]}\t{test_data.iloc[i, 2]}\t{int(prediction[0, 0])}\t{int(prediction[0, 1])}\t{int(prediction[0, 2])}\t{int(prediction[0, 3])}\t{int(prediction[0, 4])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer 训练和工作模式脚本')
    parser.add_argument('mode', choices=['train', 'work'], help='选择模式：train 或 work')
    parser.add_argument('input_file', type=str, help='输入数据文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练模式下的训练周期数')

    args = parser.parse_args()

    data = load_data(args.input_file)

    vocab_size = 1000  # 自定义词汇表大小

    if args.mode == 'train':
        model = TransformerModel(vocab_size, d_model=32, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, ff_dim=32, output_dim=5)
        train_model(model, data, data, epochs=args.epochs)
    elif args.mode == 'work':
        model = TransformerModel(vocab_size, d_model=32, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, ff_dim=32, output_dim=5)
        work_mode(model, data)
