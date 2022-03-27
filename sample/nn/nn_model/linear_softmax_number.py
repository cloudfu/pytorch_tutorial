import os
import sys
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
sys.path.append(os.getcwd())
from sample.nn.nn_model.base_model import base_model

class linear_softmax_number(base_model):
    def __init__(self, name="linear_softmax_number"):
        super().__init__(name)

        # 训练数据集 - 训练参数准备
        self.learning_rate = 0.001
        batch_size = 100

        # 生成训练数据
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

        # 测试数据集
        test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    def __init_nn__(self):
        self.input_size = 28 * 28
        num_classes = 10
        # 单一线性回归
        # self.nn_model = nn.Linear(self.input_size, num_classes)

        # 单一线性 + ReLu
        self.nn_model = nn.Sequential(
                                nn.Linear(self.input_size, num_classes))


    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.CrossEntropyLoss()  

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)  

    def __generate_data__(self):
        return torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),
                                                download=False)

    def load_batch_data(self,data_arrays,batch_size):
        return torch.utils.data.DataLoader(dataset=data_arrays,batch_size=batch_size,shuffle=True)                                     

    def train(self):
        total_step = len(self.train_loader)
        num_epochs = 3
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # form feature
                # 将 28*28 转换为 784
                images = images.reshape(-1, self.input_size)

                # Forward pass
                pred_labels = self.nn_model(images)
                loss = self.loss_function(pred_labels, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                #每轮批次训练完成看下损失率
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    def vaild(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.reshape(-1, self.input_size)
                outputs = self.nn_model(images)
                # outputs：[100,10]
                # torch.max(tensor,1)， 0:从列中获取最大值，1:从行中获取最大值
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))  

linear_softmax = linear_softmax_number()
linear_softmax.train()
# linear_softmax.load()
linear_softmax.vaild()
# linear_softmax.save()
