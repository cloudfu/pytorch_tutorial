import torch
from torch.utils import data
from torch import nn

import os, sys
sys.path.append(os.getcwd())
from sample.nn_model.linear_regression import linear_model

# 训练数据集
true_w = torch.tensor([2.2])
true_b = 3
example_count = 1000

batch_size = 10
# 对于测试数据进行小批量加载
def load_batch_data(data_arrays,batch_size,is_shuffle = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,is_shuffle)

learning_rate = 0.03

#==========================
# 神经网络 - Sequential
#==========================

# 生成线性数据模型
def generate_linear_data(w,b,example_count):

    x = torch.normal(0, 1, (example_count,len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)

    # y.reshape(-1,1)
    # UserWarning: Using a target size (torch.Size([10])) that is different to the input size (torch.Size([10, 1])). 
    # This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
    return x,y.reshape((-1, 1))

# features.shape = [1000,2]
# lables.shape = [1000, 1]
features,labels = generate_linear_data(true_w, true_b, example_count)
print(features.shape,labels.shape)

data_iter = load_batch_data((features,labels), batch_size)

# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
# 特征是设定的w权重维度，generate_data 生成的w维度是2
# TODO:true_w 可以理解为2个神经元?

linear_model = nn.Sequential(nn.Linear(len(true_w), 1))
# 可以手动设置weight and bias 也可不做设定；
# nn_model[0].weight.data.normal_(0, 0.01)
# nn_model[0].bias.data.fill_(0)

#==========================
# 神经网络 - 非线性网络
#==========================

# 生成非线性数据模型
def generate_sin_data():

    x = torch.linspace(-3.0,1.1, 2000)
    y = torch.sin(x)
    return x,y

# features,labels = generate_sin_data()
# print(features.shape,labels.shape)
# data_iter = load_batch_data((features,labels), batch_size)

# class PolynomialNet(nn.Module):
#     def __init__(self):
#         super(PolynomialNet, self).__init__()
#         self.a = torch.nn.Parameter(torch.randn(()))
#         self.b = torch.nn.Parameter(torch.randn(()))
#         self.c = torch.nn.Parameter(torch.randn(()))

#     def forward(self, x):
#         return self.a * x ** 2 + self.b * x + self.c

#     def toString(self):
#         return f'y = {self.a.item()} x**2 + {self.b.item()} x + {self.c.item()}'


#==========================
# 初始化各类数据模型
#==========================

# 定义神经模型
nn_model = linear_model

# 定义损失函数
loss_function = nn.MSELoss()

# 定义优化器:小批量随机梯度下降算法
optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

# 开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for features_iter, labels_iter in data_iter:
        prod_label = nn_model(features_iter)
        loss = loss_function(prod_label, labels_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #每轮批次训练完成看下损失率
    epoch_loss = loss_function(nn_model(features),labels)
    print(f'epoch {epoch + 1}, loss {epoch_loss:f}')

w = nn_model[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = nn_model[0].bias.data
print('b的估计误差：', true_b - b)
# print(nn_model.toString())