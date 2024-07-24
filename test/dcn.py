import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成示例2D数据
X, y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)

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

input_dim = X.shape[1]
encoding_dim = 2
autoencoder = Autoencoder(input_dim, encoding_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# 训练自动编码器
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    perm = np.random.permutation(X_tensor.size(0))
    X_tensor = X_tensor[perm]
    
    for i in range(0, X_tensor.size(0), batch_size):
        batch = X_tensor[i:i + batch_size]
        encoded, decoded = autoencoder(batch)
        loss = criterion(decoded, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 提取编码后的数据
encoded_data, _ = autoencoder(X_tensor)
encoded_data = encoded_data.detach().numpy()

# 使用 K-means 聚类编码后的数据
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(encoded_data)
labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='viridis')
plt.title('Encoded Data with K-means Clustering')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()
