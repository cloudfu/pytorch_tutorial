import os
import sys
import torch
from torch import nn
import torchvision
from torchvision.transforms import transforms
sys.path.append(os.getcwd())
from sample.nn.nn_model.base_model import base_model

class ConvNet(nn.Module):
    def __init__(self,num_classes=10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(

            # in_channels：输入的通道数目 【必选】
            # out_channels： 输出的通道数目 【必选】
            # kernel_size：卷积核的大小，类型为int 或者元组，当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。【必选】
            # stride： 卷积每次滑动的步长为多少，默认是 1 【可选】
            # padding： 设置在所有边界增加 值为 0 的边距的大小（也就是在feature map 外围增加几圈 0 ），例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5 。即在外围加了一圈 0 。【可选】
            # dilation：控制卷积核之间的间距（什么玩意？请看例子）【可选】
            # Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),

            # 归一化
            nn.BatchNorm2d(16),

            # 激活函数
            nn.ReLU(),

            # 池化层，进行下采样
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # MPL 全连接层    
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class cnn_base(base_model):
    def __init__(self, name="ConvNet"):
        super().__init__(name)

        # 生成训练数据
        batch_size = 100
        self.learning_rate = 0.001

        self.train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                train=True, 
                                                transform=transforms.ToTensor(),
                                                download=False)

        self.test_dataset = torchvision.datasets.MNIST(root='./data/',
                                                train=False, 
                                                transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

    def __init_nn__(self):
        # 线性神经模型
        self.nn_model = ConvNet()

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.CrossEntropyLoss()

    def __init_optimizer__(self):
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)

    def train(self):
        num_epochs = 5
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Forward pass
                outputs = self.nn_model(images)
                loss = self.loss_function(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    def test(self):
        # Test the model
        self.nn_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                outputs =  self.nn_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))    


cnn = cnn_base()
cnn.train()
# cnn.save()
# cnn.load()
# cnn.test()
