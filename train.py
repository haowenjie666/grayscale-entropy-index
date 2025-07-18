import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt


# 自定义Dataset类
class TifDataset(Dataset):
    def __init__(self, image_dir, excel_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform  # 存储图像的转换方法

        # 读取Excel文件
        self.data = pd.read_excel(excel_file)
        self.image_files = self.data['ImageID']  # 使用ImageID列作为图像文件名
        self.labels = self.data['EntropyValue']  # 使用EntropyValue列作为标签

        # 标签最小值和最大值
        self.label_min = 5
        self.label_max = 18

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)  # 打开tif图像
        image = np.array(image)  # 转换为numpy数组
        image = image.astype(np.float32)  # 转换为float32

        # 归一化处理，将16位灰度图像的像素值从[0, 65535]映射到[0, 1]
        image /= 65535.0

        # 保持单通道图像（不转换为RGB）
        image = np.expand_dims(image, axis=0)  # (1, 976, 776)

        # 转换为PIL图像并且维度符合ResNet的输入要求
        image = Image.fromarray(image[0])  # 转换为PIL图像

        if self.transform:
            image = self.transform(image)

        # 使用Min-Max归一化处理标签
        normalized_label = (self.labels[idx] - self.label_min) / (self.label_max - self.label_min)

        return image, normalized_label


# 数据增强与转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化图像
])

# 数据路径和Excel文件路径
image_dir = r'G:\郝文杰\熵\slice-3dprint-15ep'
excel_file = r'G:\郝文杰\熵\updated_entropy_results-slice-3dprint-15ep(35138, 52491)a.xlsx'

# 创建数据集
dataset = TifDataset(image_dir=image_dir, excel_file=excel_file, transform=transform)


train_size = int(790)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 使用ResNet50模型，不加载预训练模型
model = models.resnet50(pretrained=False)  # 不使用预训练权重

# 修改第一层卷积层，适应单通道输入
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# 修改最后的全连接层，使其适应回归任务
model.fc = nn.Linear(in_features=2048, out_features=1)  # 输出为1个数字

# 将模型放到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)

# 设置损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# 训练模型
num_epochs = 30
train_losses = []
test_losses = []

import os # Make sure to import the 'os' module at the top of your file

# ... (all your other code for setting up the model, etc.)


best_test_loss = float('inf')
best_model_save_path = r'G:\haowenjie\shang\weights\resnet50_best_model.pth'

# --- 新增代码: 确保保存目录存在 ---
# 1. 从完整文件路径中获取目录路径
model_save_dir = os.path.dirname(best_model_save_path)
# 2. 创建目录 (如果目录已存在，exist_ok=True会防止报错)
os.makedirs(model_save_dir, exist_ok=True)
print(f"Ensured directory exists: {model_save_dir}")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 确保标签形状为(batch_size, 1)并且数据类型为float32
        labels = labels.unsqueeze(1).float()

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # 每个epoch结束后在测试集上评估
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 预测
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # 反归一化并打印预测值和真实值 (可以选择性保留或注释掉这部分，避免过多打印)
            predicted = outputs.item() * (18 - 5) + 5  # 反归一化
            actual = labels.item() * (18 - 5) + 5  # 反归一化
            # print(f"Predicted: {predicted:.4f}, Actual: {actual:.4f}, sub: {predicted - actual:.4f}")

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # --- 新增代码: 检查并保存最佳模型 ---
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), best_model_save_path)
        print(f"New best model saved! Loss: {best_test_loss:.4f}")
    # --- 代码修改结束 ---

    scheduler.step(avg_train_loss) # 通常在验证损失上调用scheduler

# --- 移除的代码: 不再在最后保存模型 ---
# model_save_path = r'G:\郝文杰\熵\weights\resnet50_model.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved at {model_save_path}")
# --- 代码修改结束 ---
print(f"Training complete. Best model saved at {best_model_save_path} with test loss: {best_test_loss:.4f}")


# 绘制训练和测试损失图
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss")
plt.legend()
plt.show()