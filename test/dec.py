import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# 生成示例2D数据
X, y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)

# 定义自动编码器
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 定义DEC模型
class DEC(nn.Module):
    def __init__(self, input_dim, encoding_dim, n_clusters):
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, encoding_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)
    
    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        q = self.soft_assignment(encoded)
        return encoded, decoded, q
    
    def soft_assignment(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

input_dim = X.shape[1]
encoding_dim = 2
n_clusters = 3

dec = DEC(input_dim, encoding_dim, n_clusters)
criterion = nn.MSELoss()
optimizer = optim.Adam(dec.parameters(), lr=0.001)

# 初始化聚类中心
encoded_data, _ = dec.autoencoder(X_tensor)
encoded_data = encoded_data.detach().numpy()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(encoded_data)
dec.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

# 自定义目标分布函数
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# 训练DEC模型
num_epochs = 100
batch_size = 32
update_interval = 5
kl_weight = 0.1

for epoch in range(num_epochs):
    perm = np.random.permutation(X_tensor.size(0))
    X_tensor = X_tensor[perm]
    
    if epoch % update_interval == 0:
        encoded_data, _, q = dec(X_tensor)
        p = target_distribution(q.detach().numpy())
        y_pred = q.detach().numpy().argmax(1)
    
    for i in range(0, X_tensor.size(0), batch_size):
        batch = X_tensor[i:i + batch_size]
        batch_p = torch.FloatTensor(p[i:i + batch_size])
        
        encoded, decoded, q = dec(batch)
        recon_loss = criterion(decoded, batch)
        kl_loss = torch.mean(torch.sum(batch_p * torch.log(batch_p / q), dim=1))
        kl_weight = 0.1  # 调整 KL 散度损失的权重
        loss = recon_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 可视化DEC聚类结果
encoded_data, _, q = dec(X_tensor)
encoded_data = encoded_data.detach().numpy()
y_pred = q.detach().numpy().argmax(1)

plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=y_pred, cmap='viridis')
plt.title('DEC Clustering')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()
