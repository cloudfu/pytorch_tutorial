import torch
from torch import nn
import torch.nn.functional as F
from sample.nn.nn_model.base_model import base_model


class NeuralNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NeuralNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.out = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.out(x)
        return x


class linear_regression_sin_relu(base_model):
    def __init__(self, name="base_model"):
        super().__init__(name)

        # 训练数据集 - 训练参数准备
        x_region = [-3.0,1.1]
        batch_size = 10
        example_count = 1000

        # 生成训练数据
        self.features, self.labels = self.__generate_data__(example_count, x_region[0], x_region[1])
        self.data_iter = self.load_batch_data((self.features, self.labels),batch_size)


    def __init_nn__(self):
        # 线性神经模型
        self.nn_model = NeuralNet(n_feature=1, n_hidden=10, n_output=1)

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.MSELoss()

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

    def __generate_data__(self,example_count,start,end):
        x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
        y = x.pow(2) + 0.2*torch.rand(x.size())  
        return x,y

    def train(self):
        # 开始进行训练
        num_epochs = 200
        for epoch in range(num_epochs):
            for features_iter, labels_iter in self.data_iter:
                prod_label = self.nn_model(features_iter)
                loss = self.loss_function(prod_label, labels_iter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            #每轮批次训练完成看下损失率
            epoch_loss = self.loss_function(self.nn_model(self.features),self.labels)
            print(f'epoch {epoch + 1}, loss {epoch_loss:f}')


        
