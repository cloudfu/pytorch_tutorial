#encoding:utf-8
import torch

# 假设是时间步T1的输出
T1 = torch.rand(3,2)
# 假设是时间步T2的输出
T2 = torch.rand(3,2)


a = torch.stack((T1,T2),dim=0)
b = torch.stack((T1,T2),dim=1)
c = torch.stack((T1,T2),dim=2)

print(T1)
print(T2)
# tensor([[0.9546, 0.9240],
#         [0.7856, 0.6375],
#         [0.9728, 0.5644]])
# tensor([[0.2520, 0.3435],
#         [0.9457, 0.9533],
#         [0.6925, 0.5882]])

print(a.shape,a,b.shape,b,c.shape,c)
# torch.Size([2, 3, 2]) 
# tensor([[[0.9546, 0.9240],
#          [0.7856, 0.6375],
#          [0.9728, 0.5644]],

#         [[0.2520, 0.3435],
#          [0.9457, 0.9533],
#          [0.6925, 0.5882]]]) 
# 
# torch.Size([3, 2, 2]) 
# tensor([[[0.9546, 0.9240],
#          [0.2520, 0.3435]],

#         [[0.7856, 0.6375],
#          [0.9457, 0.9533]],

#         [[0.9728, 0.5644],
#          [0.6925, 0.5882]]]) 
# 
# torch.Size([3, 2, 2]) 
# tensor([[[0.9546, 0.2520],
#          [0.9240, 0.3435]],

#         [[0.7856, 0.9457],
#          [0.6375, 0.9533]],

#         [[0.9728, 0.6925],
#          [0.5644, 0.5882]]])
