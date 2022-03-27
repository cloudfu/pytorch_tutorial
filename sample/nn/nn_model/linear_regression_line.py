import torch
from torch import nn
from sample.nn.nn_model.base_model import base_model


class linear_regression_line(base_model):
    def __init__(self, name="base_model"):
        super().__init__(name)

        # 训练数据集 - 训练参数准备
        self.true_w = torch.tensor([-2.2, 3.14])
        self.true_b = 3
        batch_size = 10
        example_count = 1000

        # 生成训练数据
        self.features,self.labels = self.__generate_data__(example_count, self.true_w, self.true_b)
        self.data_iter = self.load_batch_data((self.features,self.labels), batch_size)


    def __init_nn__(self):
        # 线性神经模型
        self.nn_model = nn.Sequential(nn.Linear(2, 1))

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.MSELoss()

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

    def __generate_data__(self,example_count,w,b):
        x = torch.normal(0, 1, (example_count,len(w)))
        y = torch.matmul(x, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return x, y.reshape((-1, 1))

    def train(self):
        num_epochs = 3
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

        # 训练完成打印变量预估值
        w = self.nn_model[0].weight.data
        print('w的估计误差：', self.true_w - w.reshape(self.true_w.shape))
        b = self.nn_model[0].bias.data
        print('b的估计误差：', self.true_b - b)
