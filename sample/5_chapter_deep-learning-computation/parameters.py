import torch
from torch import matmul, nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(12, 4))
net(X)

# =======================
# 参数访问 全局
# =======================
# region 
print("state_dict:" + "-"*100)
print(net[1])
print(net[2].state_dict())
print(net[2].state_dict().items())

# state_dict 参数访问方式
print("state_dict_parameters:" + "-"*100)
print(net.state_dict()['2.bias'].data)

# 访问参数 bias
print("bias:" + "-"*100)
print(net[2].bias)
print(net[2].bias.data)

# 访问参数 weight
print("weight:" + "-"*100)
print(net[2].weight)
print(net[2].weight[0,1].data)
print(net[2].weight.grad)

# 一次性访问所有参数
print("named_parameters:" + "-"*100)
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print("-"*100)
# endregion

# =======================
# 从嵌套块收集参数
# =======================
# region
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    # nn.Sequential()可以获取到net进行收到添加net layout 和 net block
    net = nn.Sequential()
    for i in range(4):
        # 网络层可以嵌套module 通过add_module方法
        net.add_module(f'block {i}', block1())
    return net

block_net = block2()
for layer in block_net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
print("-"*100)

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)

# 打印每个网络层信息，但是无法挖掘block包含的layout
for layer in rgnet:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
print("-"*100)
# endregion

# =======================
# 参数初始化
# =======================
# 总结：
# 输入参数：有2个维度，行维度作为batch_size 作为样本数据（输入行从头到尾不做变化，nn数据也是这个batch_size）；
#           列数可以看做权重weight（或者维度）和隐藏层的weight进行点积，并作为下一层的输入；
#           最终输出的结果行和输入参数相同，即为batch_size，类似于 torch.matmul 第一个矩阵行决定输出行，第二个矩阵列决定输出列，列又作为下个隐藏层的输入；
# 计算过程：X:[12,4]
#                   Linear(in_features=4, out_features=8, bias=True)
#                   torch.matmul(X:[12,4], net[0].weight:[4,8]) = X:[12,4]
#           
# =======================

# 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，
# 且将偏置参数设置为0。
print("-"*100)
print("-"*100)
X = torch.rand(size=(12, 4))
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# 输入参数需要符合矩阵点乘 matmul 转置规则， 输入列和输出行相符，这样才可以matrix dot
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net)
print("input:" + str(X.shape))
# print(X)
# nn.Linear(4, 8) 
print("net[0].weight:"+str(net[0].weight.data[1,1]))
print("net[0].weight:"+str(net[0].weight[1,1].data))
# nn.Linear(8, 1)
print("net[1].weight:"+str(net[2].weight.data.shape))
print("output:" + str(net(X).data.shape))
print("-"*100)
print("-"*100)
