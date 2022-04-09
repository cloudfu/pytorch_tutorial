
# a = torch.arange(12).reshape(2,2,3)
# b = torch.arange(12).reshape(2,3,2)

# print(a)
# print(b)

# c = torch.matmul(a,b)
# print(c.shape)

# ====================
# 一维矩阵乘法
# ====================
from numpy import matrix
import torch


# mul 这两必须列数相同，
# 多维数组可以mul一维，这个时候一维数据进行广播扩展
# A = torch.tensor([[1,2,3],[1,2,3]])
# B = torch.tensor([[1,2,3],[1,2,3]])
# C = torch.rand(size=(2,3,3))
# D = torch.rand(size=(2,3,3))
# a = torch.tensor([1,2,3])
# b = torch.tensor([1,2,3])
# c = torch.tensor([1,2,3])

# print(torch.mul(a,b))
# print(torch.mul(A,a))
# print(torch.mul(A,B))
# print(torch.mul(C,D))
# print(torch.mul(C,b))

# matmul
A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
AT = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
B = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

a = torch.tensor([1,2,3])
b = torch.tensor([1,2])

# 维度相同的矩阵，进行转置之后可以点积matmul
print(torch.matmul(A,AT.T)) # pass

# 矩阵 matmul 向量
# 1.** 矩阵在左，列向量在右，矩阵的列数和列向量的维数必须相等
# 2.矩阵和向量相乘的结果也是一个向量
# 3.矩阵的行数就是最终结果输出的列向量的维数 
# 4.乘法的规则如上所示，就是矩阵的每行和列向量进行对应元素分别相乘后相加
print(torch.matmul(A,a)) # pass


# 向量 matmul 矩阵
# 1.项目个数需要和矩阵行数相等
print(torch.matmul(a,B)) # pass

# 上述两种情况其实都符合  matrix_A.col = matrix_B.row 规则体系
# matrix 和 tensor matmul 都是向量
# matrix matmul tensor
#     将tensor 按列输出排列，此时可以将 matrix.col = tensor.row 进行比对
#     [1,2,3]   [[1],           [[14],
#     [3,4,5]    [2],    =       [32],
#     [6,7,8]    [3]]            [50]]

# tensor matmul matrix 
# 将tensor 直接输出，按照matmul 规则，tensor.col = matrix.row 输出(1,matrix.col) (1,3)
#     [1,2,3]     [1,2,3]
#     NULL        [1,2,3]
#     NULL        [1,2,3]



# ====================
# 二维矩阵的乘法
# ====================
# 仔细观察这个计算公式，我们总结出以下的一些要求和规律：
# 1.左边矩阵的列数要和右边矩阵的行数相等 
# 2.左边矩阵的行数决定了结果矩阵的行数 
# 3 右边矩阵的列数决定了结果矩阵的列数
# matmul 等价于 matrix dot matrix.T  此时两个矩阵的行列相等
# print("matrix matmul matrix")
# print("-"*40)
# A = torch.tensor([[3,5,2],[3,1,2]]) #(2,3)
# B = torch.tensor([[2,1],[1,4],[2,2]]) #(3,2)
# C = torch.tensor([[2,1,2],[1,4,2]])

# print(A.matmul(B))
# print(A.matmul(C.T))