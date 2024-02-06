import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, :3].astype(str).tolist()
        labels = self.data.iloc[idx, 3:].astype(int).tolist()
        
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        
        # Adjust labels shape to [num_labels]
        labels = torch.tensor(labels)
        
        inputs['labels'] = labels
        
        return inputs


def train_model(train_data, model, tokenizer, max_length, epochs, learning_rate, checkpoint_path=None):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("从检查点加载模型。")

    for epoch in range(epochs):
        for batch in train_data:
            optimizer.zero_grad()
            inputs = batch['input_ids'].squeeze(0)
            labels = batch['labels']
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f'第 {epoch+1}/{epochs} 轮, 损失: {loss.item()}')

        # 保存检查点
        if checkpoint_path is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)


if __name__ == "__main__":
    # 选择模式
    mode = input("输入模式 (train/work): ")

    if mode == "train":
        # 读取训练数据的 CSV 文件
        csv_file_path = input("输入训练数据的 CSV 文件路径: ")
        df_train = pd.read_csv(csv_file_path, header=None, names=['text1', 'text2', 'text3', 'label1', 'label2', 'label3', 'label4', 'label5'])
        
        # 预处理数据
        df_train.fillna("no", inplace=True)
        df_train[['label1', 'label2', 'label3', 'label4', 'label5']] = df_train[['label1', 'label2', 'label3', 'label4', 'label5']].fillna(0).astype(int)

        # 初始化 tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_length = 50  # 根据数据调整

        # 准备数据集
        train_data, val_data = train_test_split(df_train, test_size=0.2, random_state=42)
        train_dataset = CustomDataset(train_data, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        # 初始化模型
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

        # 训练
        train_model(train_loader, model, tokenizer, max_length, epochs=3, learning_rate=1e-5, checkpoint_path='trained_model')

    elif mode == "work":
        # 读取工作数据的 CSV 文件
        csv_file_path = input("输入工作数据的 CSV 文件路径: ")
        df_work = pd.read_csv(csv_file_path, header=None, names=['text1', 'text2', 'text3'])

        # 预处理数据
        df_work.fillna("no", inplace=True)

        # 初始化 tokenizer
        tokenizer = BertTokenizer.from_pretrained('trained_model')
        max_length = 50  # 根据数据调整

        # 初始化模型
        model = BertForSequenceClassification.from_pretrained('trained_model')

        # 工作模式
        model.eval()

        with torch.no_grad():
            for i in range(len(df_work)):
                text = df_work.iloc[i, :3].astype(str).tolist()
                inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                outputs = model(**inputs)
                predicted_labels = outputs.logits[0].argmax().tolist()

                print(f"输入: {text}, 预测输出: {predicted_labels}")

    else:
        print("无效模式。请键入 'train' 或 'work'。")
