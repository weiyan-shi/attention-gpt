import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载数据
file1 = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku\\ClusteringResults_new.csv'
file2 = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku\\ClusteringResults_new.csv'
file3 = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku\\ClusteringResults_new.csv'

df1 = pd.read_csv(file1)
df1 = df1.dropna()
df2 = pd.read_csv(file2)
df2 = df2.dropna()
df3 = pd.read_csv(file3)
df3 = df3.dropna()


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

# 归一化特征并编码标签
def preprocess_data(df):
    X = df[features].values
    y = df['Patient Type'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return X_scaled, y_encoded

X1, y1 = preprocess_data(df1)
X2, y2 = preprocess_data(df2)
X3, y3 = preprocess_data(df3)

# 将数据转换为张量
def prepare_data(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)

train_loader_A = prepare_data(X1, y1)
train_loader_B = prepare_data(X2, y2)
train_loader_C = prepare_data(X3, y3)

train_loaders = [train_loader_A, train_loader_B, train_loader_C]

import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 定义一个简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 聚合模型参数
def average_models(global_model, client_models):
    global_state_dict = global_model.state_dict()
    for k in global_state_dict.keys():
        global_state_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_state_dict)

# 训练单个客户端模型
def train_local_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            
            # 检查NaN值并过滤
            if torch.isnan(outputs).any():
                print("NaN detected in outputs")
                continue
            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            probs = outputs.detach().numpy()
            preds = (outputs >= 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return accuracy, precision, recall, f1, auc

# 初始化
input_dim = len(features)  # 输入维度为特征数量

# 创建全局模型
global_model = SimpleMLP(input_dim)

# 创建三个客户端模型
client_models = [copy.deepcopy(global_model) for _ in range(3)]

# 定义损失函数
criterion = nn.BCELoss()

# 联邦学习训练过程
num_epochs = 10
local_epochs = 2  # 每轮联邦学习中的本地训练轮数

for epoch in range(num_epochs):
    # 在每个客户端上训练本地模型
    for i, model in enumerate(client_models):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_local_model(model, train_loaders[i], criterion, optimizer, epochs=local_epochs)
    
    # 聚合客户端模型参数到全局模型
    average_models(global_model, client_models)
    
    # 更新客户端模型为聚合后的全局模型
    client_models = [copy.deepcopy(global_model) for _ in range(3)]
    
    # 评估全局模型
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    for loader in train_loaders:
        accuracy, precision, recall, f1, auc = evaluate_model(global_model, loader)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Accuracy: {sum(accuracies)/len(accuracies):.4f}')
    print(f'Precision: {sum(precisions)/len(precisions):.4f}')
    print(f'Recall: {sum(recalls)/len(recalls):.4f}')
    print(f'F1 Score: {sum(f1s)/len(f1s):.4f}')
    print(f'AUC: {sum(aucs)/len(aucs):.4f}')

print('Finished Federated Learning')
