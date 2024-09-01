import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import time

# 读取CSV文件
file_paths = [
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_10.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults_new.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_new.csv'
]

# 要保留的列
columns_to_keep = [
    'Patient Type',
    'KMeans SC', 'KMeans CH', 'KMeans DB', 'KMeans CSL', 'KMeans DI', 'KMeans DB*', 'KMeans GD33', 'KMeans PB', 'KMeans PBM', 'KMeans STR',
    'KMedoids SC', 'KMedoids CH', 'KMedoids DB', 'KMedoids CSL', 'KMedoids DI', 'KMedoids DB*', 'KMedoids GD33', 'KMedoids PB', 'KMedoids PBM', 'KMedoids STR',
    'Agglomerative SC', 'Agglomerative CH', 'Agglomerative DB', 'Agglomerative CSL', 'Agglomerative DI', 'Agglomerative DB*', 'Agglomerative GD33', 'Agglomerative PB', 'Agglomerative PBM', 'Agglomerative STR',
    'BIRCH SC', 'BIRCH CH', 'BIRCH DB', 'BIRCH CSL', 'BIRCH DI', 'BIRCH DB*', 'BIRCH GD33', 'BIRCH PB', 'BIRCH PBM', 'BIRCH STR',
    'DBSCAN SC', 'DBSCAN CH', 'DBSCAN DB', 'DBSCAN CSL', 'DBSCAN DI', 'DBSCAN DB*', 'DBSCAN GD33', 'DBSCAN PB', 'DBSCAN PBM', 'DBSCAN STR',
    'OPTICS SC', 'OPTICS CH', 'OPTICS DB', 'OPTICS CSL', 'OPTICS DI', 'OPTICS DB*', 'OPTICS GD33', 'OPTICS PB', 'OPTICS PBM', 'OPTICS STR',
    'GMM SC', 'GMM CH', 'GMM DB', 'GMM CSL', 'GMM DI', 'GMM DB*', 'GMM GD33', 'GMM PB', 'GMM PBM', 'GMM STR'
]

# 合并CSV文件
dfs = [pd.read_csv(file) for file in file_paths]
results_df = pd.concat(dfs, join='outer', ignore_index=True)

# 保留指定的列
results_df = results_df[columns_to_keep]

# 删除包含缺失值的行
df = results_df.dropna()

# 提取特征和标签
features = [
    'KMeans SC', 'KMeans CH', 'KMeans DB', 'KMeans CSL', 'KMeans DI', 'KMeans DB*', 'KMeans GD33', 'KMeans PB', 'KMeans PBM', 'KMeans STR',
    'KMedoids SC', 'KMedoids CH', 'KMedoids DB', 'KMedoids CSL', 'KMedoids DI', 'KMedoids DB*', 'KMedoids GD33', 'KMedoids PB', 'KMedoids PBM', 'KMedoids STR',
    'Agglomerative SC', 'Agglomerative CH', 'Agglomerative DB', 'Agglomerative CSL', 'Agglomerative DI', 'Agglomerative DB*', 'Agglomerative GD33', 'Agglomerative PB', 'Agglomerative PBM', 'Agglomerative STR',
    'BIRCH SC', 'BIRCH CH', 'BIRCH DB', 'BIRCH CSL', 'BIRCH DI', 'BIRCH DB*', 'BIRCH GD33', 'BIRCH PB', 'BIRCH PBM', 'BIRCH STR',
    'DBSCAN SC', 'DBSCAN CH', 'DBSCAN DB', 'DBSCAN CSL', 'DBSCAN DI', 'DBSCAN DB*', 'DBSCAN GD33', 'DBSCAN PB', 'DBSCAN PBM', 'DBSCAN STR',
    'OPTICS SC', 'OPTICS CH', 'OPTICS DB', 'OPTICS CSL', 'OPTICS DI', 'OPTICS DB*', 'OPTICS GD33', 'OPTICS PB', 'OPTICS PBM', 'OPTICS STR',
    'GMM SC', 'GMM CH', 'GMM DB', 'GMM CSL', 'GMM DI', 'GMM DB*', 'GMM GD33', 'GMM PB', 'GMM PBM', 'GMM STR'
]

X = df[features]
y = df['Patient Type']

# 使用LabelEncoder将标签转换为数值类型
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = MLP()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train.float())
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    # 每10个epoch打印一次loss
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    start_time = time.time()
    y_pred = model(X_test).squeeze()
    y_pred_label = (y_pred >= 0.5).int()
    end_time = time.time()

# 计算准确率和AUC
accuracy = accuracy_score(y_test, y_pred_label)
auc = roc_auc_score(y_test, y_pred.numpy())

print(f'Accuracy: {accuracy:.4f}')
print(f'AUC: {auc:.4f}')
print(f'Prediction execution time: {end_time - start_time} seconds')
print(classification_report(y_test, y_pred_label))
