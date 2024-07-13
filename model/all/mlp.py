import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
# 读取CSV文件
file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\ClusteringResults_new.csv'


# 保留指定的列
df = pd.read_csv(file)

# 删除包含缺失值的行
df = df.dropna()

# 提取特征和标签
features = [
    'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score',
    'Agglomerative Silhouette Score', 'Agglomerative CH Score', 'Agglomerative DB Score',
    'KMedoids Silhouette Score', 'KMedoids CH Score', 'KMedoids DB Score',
    'OPTICS Silhouette Score', 'OPTICS CH Score', 'OPTICS DB Score'
]

X = df[features].values
y = df['Patient Type'].values

# 将标签编码为数值
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 特征归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(21, 64)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64,21)
        self.fc5 = nn.Linear(21, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    running_loss /= len(train_loader)
    # scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Val Loss: {val_loss}")

    # 保存验证集损失最小的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_mlp_model.pth')

# 加载最佳模型
loaded_model = MLP()
loaded_model.load_state_dict(torch.load('best_mlp_model.pth'))

# 测试模型性能
loaded_model.eval()
y_true = []
y_pred = []
y_pred_proba = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = loaded_model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
        y_pred_proba.extend(probabilities.numpy())

# 计算评估指标
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)

print(f"Accuracy: {accuracy}")
print(report)
print(f"AUC: {auc}")
