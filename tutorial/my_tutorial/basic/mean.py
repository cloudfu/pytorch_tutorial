import torch

x = torch.arange(12).view(4,3)
'''
注意：在这里使用的时候转一下类型，否则会报RuntimeError: Can only calculate the mean of floating types. Got Long instead.的错误。
查看了一下x元素类型是torch.int64,根据提示添加一句x=x.float()转为tensor.float32就行
'''
x = x.float()
x_mean = torch.mean(x)
print(x_mean)
# 去掉第一维度，（4,3）保留3
x_mean0 = torch.mean(x,dim=0,keepdim=True)

# 去掉第二维度，（4,3）保留4
x_mean1 = torch.mean(x,dim=1,keepdim=True)

print('x:')
print(x)
# x:
# tensor([[ 0.,  1.,  2.], 
#         [ 3.,  4.,  5.], 
#         [ 6.,  7.,  8.], 
#         [ 9., 10., 11.]])

print('x_mean0:')
# tensor(5.5000)

print(x_mean0)
# x_mean0:
# tensor([[4.5000, 5.5000, 6.5000]])

print('x_mean1:')
print(x_mean1)
# x_mean1:
# tensor([[ 1.],
#         [ 4.],
#         [ 7.],
#         [10.]])

print('x_mean:')
print(x_mean)

