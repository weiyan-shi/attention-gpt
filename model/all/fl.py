import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import syft as sy  # 引入Syft库

# 定义全局模型
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

# 初始化hook
hook = sy.TorchHook(torch)

# 假设数据集A和B只有注视点数据，数据集C有注视点和头像刺激数据
input_dim = 20  # 假设输入维度为20

# 创建虚拟工作者
workerA = sy.VirtualWorker(hook, id="workerA")
workerB = sy.VirtualWorker(hook, id="workerB")
workerC = sy.VirtualWorker(hook, id="workerC")

# 创建数据集并将它们分配给不同的工作者
X_train_A = torch.tensor(X_train_A, dtype=torch.float32)
y_train_A = torch.tensor(y_train_A, dtype=torch.float32).unsqueeze(1)
X_train_B = torch.tensor(X_train_B, dtype=torch.float32)
y_train_B = torch.tensor(y_train_B, dtype=torch.float32).unsqueeze(1)
X_train_C = torch.tensor(X_train_C, dtype=torch.float32)
y_train_C = torch.tensor(y_train_C, dtype=torch.float32).unsqueeze(1)

# 创建DataLoader
train_dataset_A = TensorDataset(X_train_A, y_train_A)
train_loader_A = DataLoader(train_dataset_A, batch_size=32, shuffle=True)

train_dataset_B = TensorDataset(X_train_B, y_train_B)
train_loader_B = DataLoader(train_dataset_B, batch_size=32, shuffle=True)

train_dataset_C = TensorDataset(X_train_C, y_train_C)
train_loader_C = DataLoader(train_dataset_C, batch_size=32, shuffle=True)

# 将数据集发送给对应的工作者
train_loader_A = sy.FederatedDataLoader(train_loader_A.federate((workerA,)), batch_size=32, shuffle=True)
train_loader_B = sy.FederatedDataLoader(train_loader_B.federate((workerB,)), batch_size=32, shuffle=True)
train_loader_C = sy.FederatedDataLoader(train_loader_C.federate((workerC,)), batch_size=32, shuffle=True)

# 创建模型实例
model = SimpleMLP(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 联邦学习训练过程
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader_A):
        # 将模型发送到工作者A
        model.send(inputs.location)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 将模型拉回到本地
        model.get()
        
        running_loss += loss.get().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader_A):.4f}')

print('Finished Training')
