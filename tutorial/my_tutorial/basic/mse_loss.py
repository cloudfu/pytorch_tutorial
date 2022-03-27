import torch
from torch import nn

# torch.nn.MSELoss(size_average=None, reduce=None, reduction: str = 'mean')
# size_average和reduce在当前版本的pytorch已经不建议使用了，只设置reduction就行了。
# reduction的可选参数有：'none' 、'mean' 、'sum'
# reduction='none'：求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。
# reduction='mean'：求所有对应位置差的平方的均值，返回的是一个标量。
# reduction='sum'：求所有对应位置差的平方的和，返回的是一个标量。

x = torch.Tensor([[1, 2, 3],
                  [2, 1, 3],
                  [3, 1, 2]])

y = torch.Tensor([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])


# reduction='none'：求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。
criterion1 = nn.MSELoss(reduction='none')
loss1 = criterion1(x, y)
print(loss1)

# reduction='mean'：求所有对应位置差的平方的均值，返回的是一个标量。
criterion2 = nn.MSELoss(reduction='mean')
loss2 = criterion2(x, y)
print(loss2)

# reduction='sum'：求所有对应位置差的平方的和，返回的是一个标量。
criterion3 = nn.MSELoss(reduction='sum')
loss3 = criterion3(x, y)
print(loss3)

