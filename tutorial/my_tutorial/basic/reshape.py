import torch


# a = torch.rand(1,3,4)
# print("a的shape:",a.shape)
# b = torch.reshape(a,((4,3,1)))
# print("b:",b)
# print("b的shape:",b.shape)


a = torch.rand(2,2,3)

# 输入维度必须是2维
# torch.Size([12])    
b = torch.reshape(a,(-1,))

# 将数据整理为2维，第二维度是3，第一维度即为4
# torch.Size([4, 3]) 
c = torch.reshape(a,(-1,3))

# 所有维度相乘和初始维度相符即可，注意不是相加；
# torch.Size([4, 3, 1])
d = torch.reshape(a,((-1,3,1)))
# print(d)

# torch.Size([12, 1, 1, 1])
e = torch.reshape(a,((-1,1,1,1)))

# torch.Size([1, 3, 4])
f = torch.reshape(a,((-1,3,4)))

print("a的shape：",a.shape)
print("b的shape：",b.shape)
print("c的shape：",c.shape)
print("d的shape:",d.shape)
print("e的shape:",e.shape)
print("f的shape:",f.shape)
