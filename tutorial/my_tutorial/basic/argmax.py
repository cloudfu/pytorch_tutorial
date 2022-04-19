import torch

x = torch.randn(2,4,5)
# 着眼于 dim=对应标签号，之后记性扣减，
# dim=0 剩余 4,5 输出就是4,5
# dim=1 剩余 2,5 输出就是2,5
# dim=2 剩余 2,4 输出就是2,4
# 输出的维度即为消失的dim指向维度
print(x.shape)
print(x)

# 将dim=0合并，计算第二列和第三列进行softmax计算
# print(x.argmax(dim=0))
# > tensor([[0, 0, 0, 1, 1],
#         [0, 0, 0, 1, 1],
#         [1, 0, 1, 1, 0],
#         [1, 1, 1, 0, 1]])
print(x.argmax(dim=0))

print(x.argmax(dim=1))
# tensor([[3, 0, 0, 2, 0], 
#         [0, 2, 0, 0, 2]])

print(x.argmax(dim=2))
# > tensor([[2, 4, 1, 2],
#         [4, 3, 0, 2]])
