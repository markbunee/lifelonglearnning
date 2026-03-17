import torch
dtype = torch.float
device = torch.device("cpu")

a = torch.randn(2,3,device=device,dtype=dtype)
b = torch.randn(2,3,device=device,dtype=dtype)

print(torch.__version__)
print(torch.cuda.is_available())

\#tensor
\#张量的概念类似于 NumPy 中的数组，
\# 但是 PyTorch 的张量可以运行在不同的设备上
\#维度
\#形状
\#数据类型
\#跟numpy类似
a = torch.zeros(2,3)
b = torch.ones(2,3)
c = torch.randn(2,3)

import numpy as np
numpy_array = np.array([[1,2],[3,4]])
\#与numpy共享张量
tensor_from_numpy = torch.from_numpy(numpy_array)
\#设备挂载
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
d = torch.randn(2,3,device=device)

\#创建一个需要梯度的张量
tensorgrad = torch.tensor([1.0],requires_grad = True)

\#操作比如
tensorresult = tensorgrad * 2

\#计算梯度
tensorresult.backward()
print(tensorgrad.grad)

\#优化性能和内存
\#自动求导 基于链式法则

\#requires_grad 判断是否计算张量梯度
x = torch.randn(2,2,requires_grad=True)
print(x)

y = x + 2
z = y * y * 3
out = z.mean()

print(out)
\#定义计算图后backward 计算梯度
out.backward()
print(x.grad)

\#停止梯度计算
with torch.no_grad():
y = x * 2
\#或设定requires_grad = False

\#神经网络
\#神经网络通过调整神经元之间的连接权重来优化预测结果
\#涉及前向传播、损失计算、反向传播和参数更新
\#类型包括前馈神经网络、卷积神经网络（CNN）、
\# 循环神经网络（RNN）和长短期记忆网络（LSTM）

\#简单的神经网络
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
def __init__(self):
super(SimpleNN,self).__init__()
\#输入层到隐藏层
self.fc1 = nn.Linear(2,2)
\#隐藏层到输出层
self.fc2 = nn.Linear(2,1)

def forward(self,x):
\#relu 激活函数
x = torch.relu(self.fc1(x))
x = self.fc2(x)
return x

model = SimpleNN()

print(model)

\#向前传播 在前向传播阶段，输入数据通过网络层传递，
\# 每层应用权重和激活函数，直到产生输出。

\#计算损失 根据网络的输出和真实标签，计算损失函数的值

\#反向传播 反向传播利用自动求导技术
\# 计算损失函数关于每个参数的梯度。

\#参数更新 使用优化器根据梯度更新网络的权重和偏置。

\#迭代 重复上述过程，直到模型在训练数据上的性能达到满意的水平。

\#前向传播和损失计算
x = torch.randn(1,2)
\#model先前定义好的模型
model = SimpleNN()
output = model(x)
\#均方误差 预测与真实 常用的回归任务也的损失函数
criterion = nn.MSELoss()
\#输入值
target = torch.randn(1,1)
\#计算损失函数 均方误差
loss = criterion(output,target)
print(loss)
\#其他损失函数
\#分类
nn.CrossEntropyLoss() #交叉熵 多分类问题
nn.BCELoss() #二元交叉熵
\#回归
nn.L1Loss() #绝对误差
nn.SmoothL1Loss() #平滑版的l1
nn.MSELoss()

\#优化器
\#更新神经网络的参数
\# 减少损失函数的值
\#sgd adam
\#定义优化器
optimizer = optim.Adam(model.parameters(),lr = 0.001)
\#步骤
\#清空梯度
optimizer.zero_grad()
\#反向传播
loss.backward()
\#更新参数
optimizer.step()

\#简单的
import torch
import torch.nn as nn
import torch.optim as optim

\#定义一个简单的神经网络
class SimpleNN(nn.Module):
def __init__(self):
super(SimpleNN,self).__init__()
self.fc1 = nn.Linear(2,2)
self.fc2 = nn.Linear(2,1)

def forward(self,x):
x = torch.relu(self.fc1(x))
x = self.fc2(x)
return x
\#创建模型实例
model = SimpleNN()

\#定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

\#假设有数据
X = torch.randn(10, 2) # 10 个样本，2 个特征
Y = torch.randn(10, 1) # 10 个目标值

\#训练循环
for epoch in range(100):
optimizer.zero_grad()
output = model(X)
loss = criterion(output,Y)
loss.backward()
optimizer.step()
\#每十轮一次损失
if (epoch+1) % 10 == 0:
print(f"Epoch {epoch+1}, Loss: {loss.item()}")


\#使用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
\#将数据移动到设备
x = x.to(device)
y = y.to(device)

\#创建张量
import torch

tensor = torch.tensor([1,2,3])
\#将numpy数组转化为张量
np_array = np.array([1,2,3])
tensor = torch.from_numpy(np_array)

\#创建2d张量
tensor_2d = torch.tensor([[1,2,3],[4,5,6]])
print("2d tensor",tensor_2d)
print("shape",tensor_2d.shape)

\#创建3d张量
tensor_3d = torch.stack([tensor_2d,tensor_2d + 10,tensor_2d - 5])
tensor_4d = torch.stack([tensor_3d,tensor_3d + 10,tensor_3d - 5])
tensor_5d = torch.stack([tensor_4d,tensor_4d + 10,tensor_4d - 5])

import torch
tensor = torch.tensor([1,2,3],dtype=torch.float32)
print(tensor)
print(tensor.shape) #或者 tensor.size()
print(tensor.dtype)
print(tensor.dim())
print(tensor.device)
print(tensor.requires_grad)
print(tensor.is_cuda)
print(tensor.is_contiguous())
\#获取
single_value = torch.tensor(42)
print(single_value.item())
\#转置张量
tensor = tensor.T

\#索引和切片
print("获取第一行第一列的元素:", tensor[0, 0])
print(tensor[:,1])

\#形状变换
reshaped = tensor.view(3,2)
flattened = tensor.flatten()

\#数学操作运算
tensor_add = tensor + 10
tensor_mul = tensor * 2
tensor_sum = tensor.sum()
print(tensor_sum.item())
\#item 将张量的单个值转化为python的标量值

\#其他张量操作
tensor2 = torch.tensor([[1,2,3],[1,1,1]],dtype = torch.float32)
tensor_dot = torch.matmul(tensor,tensor2.T)

\#条件判断和筛选
mask = tensor > 3
filtered_tensor = tensor[tensor > 3]
print(filtered_tensor)

\#numpy 转换pytorch
tensor_from_numpy = torch.from_numpy(np.array([1,2,3]))
\#共享内存
numpy_array[0,0] = 100
\#pytorch 张量转化为numpy
tensor = torch.tensor([1,2,3],dtype=torch.float32)
\#tensor.numpy()转numpy
numpy_from_tensor = tensor.numpy()
\#内存共享



\#神经网络基础
'''
神经元
层

前馈神经网络 FNN
输入层的每个节点代表一个输入特征。
每个隐藏层由多个神经元组成，每个神经元通过激活函数增加非线性能力。

循环神经网络RNN
一类专门处理序列数据的神经网络，能够捕获输入数据中时间或顺序信息的依赖关系
记忆能力

'''

\#torch.nn 模块提供了各种网络层（如全连接层、卷积层等）、损失函数和优化器，
\#简单的全连接神经网络
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
def __init__(self):
super(SimpleNN,self).__init__()
self.fc1 = nn.Linear(2,2)
self.fc2 = nn.Linear(2,1)

def forward(self,x):
x = torch.relu(self.fc1(x))
x = self.fc2(x)
return x

model = SimpleNN()
print(model)

\#没有激活函数 回归
\#输出层视为未归一化的logits 外部应用sigmoid
\#多分类任务 softmax




\#常见神经网络层
'''
\##全连接层输入特征输出特征
nn.Linear(in_features, out_features)
\#2d 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
groups=
普通卷积 所有输入通道共同参与

减少运算量
深度可分离卷积 in_channels 不予其他通道共享参数
分组卷积 将通道分组


\#最大池化层 降维
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
\#激活函数 隐藏层
nn.ReLU()

\#多分类问题
nn.Softmax(dim)
'''
\#激活函数
'''
sigmoid 输出值0或1
tanh 输出值1 -1 输出层前
relu 解决梯度小时
softmax 多分类 输出层 输出概率

'''
import torch.nn.functional as F
import torch
\# # ReLU 激活
\# output = F.relu(input_tensor)

\# # Sigmoid 激活
\# output = torch.sigmoid(input_tensor)

\# # Tanh 激活
\# output = torch.tanh(input_tensor)

\# 均方误差损失
criterion = nn.MSELoss()

\# 交叉熵损失
criterion = nn.CrossEntropyLoss()

\# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()

\#优化器
import torch.optim as optim
optimizer = optim.SGD(model.parameters(),lr=0.001)
optimizer = optim.Adam(model.parameters(),lr=0.001)

\#训练数据
X = torch.randn(10, 2)
Y = torch.randn(10, 1)

for epoch in range(100):
\#设置模式未训练模式
model.train()
\#清除梯度
optimizer.zero_grad()
\#向前传播 输入数据通过层传递
output = model(x)
\#计算损失
loss = criterion(output,Y)
\#反向传播
loss.backward()
\#更新权重
optimizer.step()

if (epoch + 1) % 10 == 0:
print(f'epoch[{epoch + 1}/100],loss:{loss.item():.4f}')

\#测试与评估
model.eval()
\# 在评估过程中禁用梯度计算
X_test = torch.randn(10, 2)
Y_test = torch.randn(10, 1)
with torch.no_grad():
output = model(X_test)
loss = criterion(output,Y_test)
print(f'Test Loss:{loss.item():.4f}')


'''
神经网络类型
前馈神经网络（Feedforward Neural Networks）：数据单向流动，从输入层到输出层，无反馈连接。
卷积神经网络（Convolutional Neural Networks, CNNs）：适用于图像处理，使用卷积层提取空间特征。
循环神经网络（Recurrent Neural Networks, RNNs）：适用于序列数据，如时间序列分析和自然语言处理，允许信息反馈循环。
长短期记忆网络（Long Short-Term Memory, LSTM）：一种特殊的RNN，能够学习长期依赖关系。
'''

\#二分类网络
import torch.nn as nn

\#定义层 和 批量大小
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

\#创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0],
[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])








\#nn.Sequential 用于按顺序定义网络层。
\#pytorch提供的容器类
\#将数据输入到定义顺序堆叠多个模块
\#使用 nn.Sequential 可以减少显式定义 forward 方法的需要，
\# 因为 PyTorch 会自动根据层的顺序完成前向传播。
model = nn.Sequential(
nn.Linear(n_in, n_h),
nn.ReLU(),
nn.Linear(n_h, n_out),
nn.Sigmoid()
)

\#字典命名层的方法
from collections import OrderedDict
model = nn.Sequential(
OrderedDict([
('fc1', nn.Linear(10, 20)), # 第一层
('relu1', nn.ReLU()), # 激活函数
('fc2', nn.Linear(20, 1)) # 第二层
])
)
\#后期直接调用 model.fc1

\#循环动态生成网络层
layers = []
for i in range(3):
layers.append(nn.Linear(10,10))
layers.append(nn.ReLU())

model = nn.Sequential(*layers)

\#orderedict
\#，它与普通的字典（dict）类似，
\# 但有一个重要的区别：OrderedDict
\# 会记住元素插入的顺序

'''
从 Python 3.7 开始，
普通字典也保证了插入顺序，
因此在许多情况下，可以直接使用普通字典替代
OrderedDict。
'''

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

\# 用于存储每轮的损失值
losses = []


for epoch in range(1000):
y_pred = model(x)
loss = criterion(y_pred, y)
print(epoch,loss.item())

optimizer.zero_grad()

loss.backward()
optimizer.step()

import matplotlib.pyplot as plt
\#可视化损失函数
plt.figure(figsize=(8, 5))

plt.plot(range(1,51),losses,label = 'loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

\#最终预测值
y_pred_final = model(x).detach().numpy()
\#实际值
y_actual = y.numpy()

plt.figure(figsize=(8, 5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim

\#目标是根据点的未知分类到两个类别中
\# 生成一些随机数据
n_samples = 100
data = torch.randn(n_samples, 2) # 生成 100 个二维数据点
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1) # 点在圆内为1，圆外为0

\# 可视化数据
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

'''
前向传播：
主要任务是计算模型的输出和损失。
不涉及参数更新。
反向传播：
基于前向传播的结果（损失值），计算梯度并更新参数。
是训练模型的核心步骤。
'''

\#自定义神经网络
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)

\#训练模型
epochs = 1000
for epoch in range(epochs):
\#前向传播
outputs = model(data)
loss = criterion(outputs,labels)
\#反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()


\#数据处理和加载
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
def __init__(self,X_data,Y_data):
self.X_data = X_data
self.Y_data = Y_data

def __len__(self):
return len(self.X_data)

def __getitem__(self,idx):
x = torch.tensor(self.X_data[idx],dtype=torch.float32)
y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
return x,y

X_data = [[1, 2], [3, 4], [5, 6], [7, 8]] # 输入特征
Y_data = [1, 0, 1, 0] # 目标标签

dataset = MyDataset(X_data, Y_data)




\#dataloader 加载数据
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

for epoch in range(1):
for batch_idx,(inputs,labels) in enumerate(dataloader):
print(f'batch{batch_idx + 1}')
print(f'Inputs: {inputs}')
print(f'Labels: {labels}')

\#数据预处理和数据增强
import torchvision.transforms as transforms
from PIL import Image

\#定义数据预处理刘淑仙
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

\#加载图像
image = Image.open('path_to_image.jpg')

\#应用处理
image_tensor = transform(image)

\#图像数据增强
transform = transforms.Compose([
transforms.RandomHorizontalFlip(), # 随机水平翻转
transforms.RandomRotation(30), # 随机旋转 30 度
transforms.RandomResizedCrop(128), # 随机裁剪并调整为 128x128
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


\#加载图像数据集
import torchvision.datasets as datasets

\#下载并加载mnist数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)



\#创建dataloader
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

\#迭代训练数据集
for inputs,labels in train_loader:
print(inputs.shape)
print(labels.shape)


from torch.utils.data import Dataset, DataLoader,ConcatDataset

\#combined_dataset = ConcatDataset([dataset1,dataset2])
\#combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)



\#pytorch 线性回归
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
\#100 个样本 每个样本两个特征
X = torch.randn(100, 2)
true_w = torch.tensor([2.0,3.0])
true_b = 4.0

Y = X @ true_w + true_b + torch.randn(100, 2) * 0.01

print(X[:5])
print(Y[:5])

\#定义线性回归模型
class LinearRegressionModel(nn.Module):
def __init__(self):
super(LinearRegressionModel, self).__init__()
self.linear = nn.Linear(2, 1)

def forward(self,x):
return self.linear(x)

modle = LinearRegressionModel()

num_epochs = 1000
for epoches in range(num_epochs):
model.train()
predictions = model(X)
loss = criterion(predictions.squeeze(),Y)

with torch.no_grad(): # 评估时不需要计算梯度
predictions = model(X)

\# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()

\#卷积神经网络 基本结构

\#输入层

\#卷积层 提取特征 生成特征图

\#池化 减少特征图尺寸

\#特征提取 多个卷积和池化层组合

\#展平层 将多为特征图转换为一维向量 输入到全连接层

\#全连接层 将提取特征映射到输入类别
\#全连接层的每个神经元都与前一层的所有神经元相连，
\#用于综合特征并进行最终的分类或回归

\#分类 根据全连接层输出进行分类

\#概率分布 给出每个类别的概率

\#激活函数 引入非线性特性 实质能学习到更加复杂的模式

\#归一化层 加速训练过程和太高稳定性

\#正则化
\# 包括 Dropout、L1/L2 正则化等技术，用于防止模型过拟合。
\# 这些层可以堆叠形成更深的网络结构，以提高模型的学习能力。
\# CNN 的深度和复杂性可以根据任务的需求进行调整。
'''
卷积层
应用一组可学习的滤波器（或卷积核）在输入图像上
进行卷积操作，以提取局部特征
每个滤波器在输入图像上滑动，
生成一个特征图（Feature Map）
卷积层可以有多个滤波器，
每个滤波器生成一个特征图，
所有特征图组成一个特征图集合。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
transforms.ToTensor(), # 转为张量
transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1, 1]
])

\# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


\#定义cnn
class SimpleCNN(nn.Module):
def __init__(self):
super(SimpleCNN,self).__init__()
\#输入1 输出32 卷积核3X3
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
\#输入32 输出64 卷积核3X3
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
\#定义全连接层
\#64前一层的输入通道数
\#7 * 7 是特征图的空间尺寸（高度和宽度）。
\#64 * 7 * 7 表示展平后的输入向量长度。
\#将卷积层提取的高维特征（64 * 7 * 7）转换为一个固定大小的特征向量（128）
\#降维通过学习权重矩阵w核偏置b将高位特征投影到低维空间中
self.fc1 = nn.Linear(64 * 7 * 7, 128) # 输入大小 = 特征图大小 * 通道数
self.fc2 = nn.Linear(128, 10) # 10 个类别
\#如果一下子降维太多 考虑中间层逐步降维


def forward(self, x):
x = F.relu(self.conv1(x)) # 第一层卷积 + ReLU
x = F.max_pool2d(x, 2) # 最大池化
x = F.relu(self.conv2(x)) # 第二层卷积 + ReLU
x = F.max_pool2d(x, 2) # 最大池化
x = x.view(-1, 64 * 7 * 7) # 展平操作
\#-1 自动计算批量大小
x = F.relu(self.fc1(x)) # 全连接层 + ReLU
x = self.fc2(x) # 全连接层输出
return x

model = SimpleCNN()


criterion = nn.CrossEntropyLoss() # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # 学习率和动量


\#可视化操作
\#训练
num_epochs = 10
model.train()

for epoch in range(num_epochs):
total_loss = 0
for images,labels in train_loader:
outputs = model(images)
loss = criterion(outputs,labels)

optimizer.zero_grad() # 清空梯度
loss.backward() # 反向传播
optimizer.step() # 更新参数

total_loss += loss.item()
print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

\# 5. 模型测试
model.eval() # 设置模型为评估模式
correct = 0
total = 0


with torch.no_grad(): # 关闭梯度计算
for images, labels in test_loader:
outputs = model(images)
_, predicted = torch.max(outputs, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

\# 6. 可视化测试结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
axes[i].imshow(images[i][0], cmap='gray')
axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
axes[i].axis('off')
plt.show()

\#循环神经网络RNN
\#专门用于序列数据
\#能够捕捉时间序列或有序数据的动态信息，
\# 能够处理序列数据，如文本、时间序列或音频。

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np

class SimpleRNN(nn.Module):
def __init__(self,input_size,hidden_size,output_size):
super(SimpleRNN,self).__init__()
self.rnn = nn.RNN(input_size,hidden_size,batch_first=True)
self.fc = nn.Linear(hidden_size,output_size)

def forward(self,x):
out,_ = self.rnn(x)
out = out[:,-1,:]
out = self.fc(out)
return out

num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2

x = torch.randn(num_samples,seq_len,input_size)
y = torch.randint(0,output_size,(num_samples,))

dataset = TensorDataset(x,y)
train_loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = SimpleRNN(input_size,input_size,hidden_size=64,output_size=output_size)


with torch.no_grad():
total = 0
correct = 0
for inputs,labels in train_loader:
outputs = model(inputs)
\#下划线是对某个变量不感兴趣
_,predicted = torch.max(outputs.data,1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
print(f'Accuracy: {100*correct/total}%')




\#构建一个rnn实例

char_set = list("hello")
\#char_to_idx 将字符映射到索引（如 'h' -> 0, 'e' -> 1）。
\#idx_to_char 将索引映射回字符（如 0 -> 'h', 1 -> 'e'）。
char_to_idx = {c: i for i,c in enumerate(char_set)}
idx_to_char = {i: c for i,c in enumerate(char_set)}


input_str = "hello"
target_str = "elloh"
\#将字符串转成索引列表
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

\#转成独热编码
\#将分类变量转换为数值表示
\#n个类别的分类变量会表示为一个长度为n的向量 对应类别位置为1 其余位置为0
\#np.eye(4)
input_one_hot = np.eye(len(char_set))[input_data]

\#转换为pytorch tensor
inputs = torch.tensor(input_one_hot,dtype=torch.float32)
targets = torch.tensor(target_data,dtype=torch.long)

\#模型超参数
input_size= len(char_set)
hidden_size= 128
output_size= len(char_set)
num_epochs = 100
learning_rate = 0.01

\#定义rnn
class RNN(nn.Module):
def __init__(self,input_size,hidden_size,output_size):
super(RNN,self).__init__()
self.rnn = nn.RNN(input_size,hidden_size,batch_first=True)
self.fc = nn.Linear(hidden_size,output_size)


def forward(self,x,hidden):
out,hidden = self.rnn(x,hidden)
out = self.fc(out)
return out,hidden

model = RNN(input_size,hidden_size,output_size)

losses = []
hidden = None
for epoch in range(num_epochs):
optimizer.zero_grad()
outputs,hidden = model(inputs.unsqueeze(0),hidden)

loss = criterion(outputs.view(-1,output_size),targets)

losses.append(loss.item())


\#传统前馈网络输入层到输出层 rnn 数据会在每个时间步骤传播当前隐层状态
\#RNN 通过隐状态来记住序列中的信息。
\#输入序列

\#torch.nn.RNN
\#torch.nn.LSTM
\#torch.nn.GRU

hidden = None
for i in range(len(inputs)):
hidden = hidden.detach()


\#使用自定义数据集

'''
Dataset 是 PyTorch 中用于数据集抽象的类。
自定义数据集需要继承 torch.utils.data.Dataset 并重写以下两个方法：

'''
class MyDataset(Dataset):
def __init__(self,data,labels):
self.data = data
self.labels = labels

def __len__(self):
return len(self.data)

def __getitem__(self,index):
return self.data[index],self.labels[index]

from torch.utils.data import dataloader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
dataset = MyDataset(data, labels)

for batch_idx,(batch_data,batch_labels) in enumerate(dataloader):
print(f"batch {batch_idx}")







\#使用内置数据集

import torchvision
train_dataset = torchvision.datasets.MNIST(
root='./data',train=True,transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

data_iter = iter(train_loader)
images,labels = next(data_iter)





\#读取csv文件

import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class CSVDataset(Dataset):
def __init__(self,csv_path,transform=None):
self.data = pd.read_csv(csv_path)

def __len__(self):
return len(self.data)

def __getitem__(self,idx):
row = self.data.iloc[idx]
features =- torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)
label = torch.tensor(row.iloc[-1],dtype=torch.float32)

return features,label

\#实例化
dataset = CSVDataset("runoob_pytorch_data.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for features, label in dataloader:
print("特征:", features)
print("标签:", label)
break

\#数据转换

transform = transforms.ToTensor()
transform = transforms.Normalize()

transform = transforms.Resize((128,128))
transform = transforms.CenterCrop(128)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
\# 查看转换后的数据
for images, labels in train_loader:
print("图像张量大小:", images.size()) # [batch_size, 1, 128, 128]
break

def show_images(dataset):
\#创建画布
\#fig画布 axs为子图
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
image, label = dataset[i]
\#去掉第一个维度 gray灰度模式
axs[i].imshow(image.squeeze(0), cmap='gray') # 将 (1, H, W) 转为 (H, W)
axs[i].set_title(f"Label: {label}")
axs[i].axis('off')
plt.show()

x = torch.tensor([1,2,3])
y = torch.zeros(2,3)


\#dropout层
\#drop层 正则化技术 随机丢弃一部分神经元
\#等效模型集成
\#每次迭代都相当于训练了一个不同的子网络。
\# 最终的模型可以看作是这些子网络的集合。

\#嵌入层
\#将离散的高维稀疏数据（如词索引、类别标签等）映射到低维稠密的连续向量空间
'''
在 NLP 中，嵌入层可以将单词索引
（如 3 表示 "cat"，5 表示 "dog"）映射为一个低维向量，
使得语义相近的单词在向量空间中距离较近。