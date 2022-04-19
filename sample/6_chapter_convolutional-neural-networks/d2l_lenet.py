import torch
from torch import nn
from d2l import torch as d2l
from torchvision import transforms
import torchvision
import matplotlib.pylab as plt



net = nn.Sequential(

    # 保持输入和输出形状不发生变化：padding_size = (kernel_size -1)/2
    nn.Conv2d(1, 6, kernel_size=5, padding=2), 
    nn.Sigmoid(),
    # stride=2，输出数据/2
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 没有增加padding_size，输出 input_size-kernal_size+1=output_size
    nn.Conv2d(6, 16, kernel_size=5), 
    nn.Sigmoid(),    
    # stride=2，输出数据/2 输出：16, 5, 5
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 输出：16*5*5=400
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), 
    nn.Sigmoid(),
    nn.Linear(120, 84), 
    nn.Sigmoid(),
    nn.Linear(84, 10))

# image_size = 784
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

# 打印所有网络嵌套层
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
    
# =========================================================
# print layout info:
# =========================================================
# Conv2d output shape:     torch.Size([1, 6, 28, 28])        
# Sigmoid output shape:    torch.Size([1, 6, 28, 28])        
# AvgPool2d output shape:  torch.Size([1, 6, 14, 14])

# Conv2d output shape:     torch.Size([1, 16, 10, 10])       
# Sigmoid output shape:    torch.Size([1, 16, 10, 10])       
# AvgPool2d output shape:  torch.Size([1, 16, 5, 5]) 

# Flatten output shape:    torch.Size([1, 400])

# Linear output shape:     torch.Size([1, 120])
# Sigmoid output shape:    torch.Size([1, 120])
# Linear output shape:     torch.Size([1, 84])
# Sigmoid output shape:    torch.Size([1, 84])
# Linear output shape:     torch.Size([1, 10])
# =========================================================

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # FashionMNIS dataset url
    # http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    # http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    # http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    # http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)

    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 评估GPU精确度  [ɪˈvæljueɪt] 
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 计算预测值和标签值匹配度（精度）
# 将rediction_y 和lable_y 进行比较 True=1 / False=0 并将tensor 进行累加
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    # y_hat.shape:(256,10)
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    # 类型转换并进行True/False比较
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):

    train_loss_array = []
    train_acc_array = []
    idx = []

    # 初始化权重,避免梯度爆炸：nn.init.xavier_uniform_
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        
        # 训练损失之和，训练准确率之和，范例数
        # 生成3组动态跟踪值，训练/测试/精度
        metric = d2l.Accumulator(3)
        net.train()

        # X.shape:(256,1,28,28)
        # y.shape:(256)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            # y_hat.shape:(256,10)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                # 均方差 * 批量比对数量
                # 和标签值进行匹配并进行sum合计
                # 批量比对数量
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()

            # 训练损失数
            train_l = metric[0] / metric[2]
            # 训练精度
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_loss_array.append(train_l)
                train_acc_array.append(train_acc)
                idx.append(len(idx))
                print("train_loss:"+str(train_l))
                print("train_acc:"+str(train_acc))
                plt.cla()
                plt.plot(idx, train_loss_array)  
                plt.plot(idx, train_acc_array)  
                plt.pause(0.1)
                # animator.add(epoch + (i + 1) / num_batches,
                #              (train_l, train_acc, None))
        # # 训练精度
        # test_acc = evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))
        
    # print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
    #       f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
    #       f'on {str(device)}')




lr, num_epochs = 0.9, 10
plt.ion()
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.ioff()
plt.show()