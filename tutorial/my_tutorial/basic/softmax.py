import torch
import torch.nn as nn
# m = nn.Softmax(dim=1) #注意是沿着那个维度计算
# input = torch.randn(2,2)
# print("input:")
# print(input)
# output = m(input)
# print(output)

# #注意区分以下结果,这是两个不同size的tensor
# input1=torch.randn(2,1)
# print("input1：")
# print(input1)
# print(m(input1))
# input2=torch.randn(1,2)
# print("input2:")
# print(input2)
# print(m(input2))

# 计算出每个维度的数据大小占比

def softmax(x):
    s = torch.exp(x)
    # s:
    # tensor([[ 2.7183,  7.3891, 20.0855],
    #     [ 7.3891, 20.0855,  2.7183]])
    
    exp_sum = torch.sum(s, dim=1, keepdim=True)
    # exp_sum: 如果去掉keepdim 则输出一维数组
    # tensor([[30.1929],
    #     [30.1929]])

    val = s / exp_sum
    # tensor([[0.0900, 0.2447, 0.6652],
    #           [0.2447, 0.6652, 0.0900]])
    return val

a = torch.tensor([[1,2,3.],[2,3,1.]])
print(softmax(a))