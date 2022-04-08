# A表示均值，
# B表示标准差,
# C代表生成的数据行数，
# D表示列数，
# requires_grad=True表示对导数开始记录
# torch.normal(A, B ,size(C, D), requires_grad=True)

import torch
w = torch.normal(1, 0.02, size=(3, 1), requires_grad=True)
print(w)
# tensor([[0.9850],
#         [0.9749],
#         [1.0409]], requires_grad=True)