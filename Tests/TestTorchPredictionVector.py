import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 数据准备
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

# 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, 64)
        self.transformer = nn.Transformer(64, nhead, num_layers)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src)
        output = self.fc(output[-1, :, :])
        return output

# 训练函数
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            src = batch.permute(1, 0, 2)  # 调整输入维度
            output = model(src)
            loss = criterion(output, src[:, -1, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# 自定义collate_fn处理变长序列
def collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    padded_batch = torch.zeros((len(batch), max_len, len(batch[0][0])))
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq), :] = torch.tensor(seq)
    return padded_batch




# 数据准备
data = [[[1, 2, 3], [4, 5, 6]],
        [[7, 8], [9, 10, 11, 12]],
        [[13, 14, 15], [16, 17]]]

input_dim = 20
output_dim = 10

# 创建模型和数据加载器
model = TransformerModel(input_dim, output_dim, nhead=2, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = CustomDataset(data)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 开始训练
train(model, train_loader, optimizer, criterion, epochs=50)
