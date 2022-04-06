import torch
from d2l import torch as d2l


def corr2d(X, K):  
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    # 先拆解第0维度的数据集，拆解后就是一个2位图片和2维卷积和进行互相关预算；
    for x, k in zip(X, K):
        print(x,k)
    return sum(corr2d(x, k) for x, k in zip(X, K))

# X.shape = (2,3,3)
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

# K.shape = (2,2,2)
K = torch.tensor([
                    [[0.0, 1.0], [2.0, 3.0]], 
                    [[1.0, 2.0], [3.0, 4.0]]
                ])

print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

# def corr2d_multi_in_out_1x1(X, K):
#     c_i, h, w = X.shape
#     c_o = K.shape[0]
#     X = X.reshape((c_i, h * w))
#     K = K.reshape((c_o, c_i))
#     # 全连接层中的矩阵乘法
#     Y = torch.matmul(K, X)
#     return Y.reshape((c_o, h, w))


# X = torch.normal(0, 1, (3, 3, 3))
# K = torch.normal(0, 1, (2, 3, 1, 1))

# Y1 = corr2d_multi_in_out_1x1(X, K)
# Y2 = corr2d_multi_in_out(X, K)
# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6