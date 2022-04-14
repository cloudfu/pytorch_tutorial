import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-1, 1, 50)
y1 = 2*x + 1
y2 = 2**x + 1

# num表示的是编号，是否进行独立图像输出或者合并输出；
plt.figure(num = 3, figsize=(8, 5))  
l1, = plt.plot(x, y2,label='A1')

# 设置线条的样式,主要返回接收参数
l2, = plt.plot(x, y1, 
                color='red',  # 线条的颜色
                linewidth=1.0,  # 线条的粗细
                linestyle='--',  # 线条的样式
                label='B1'
                )

# 设置取值参数范围
plt.xlim((-1, 2))  # x参数范围
plt.ylim((1, 3))   # y参数范围

# 设置坐标轴的步进和Labels
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)

# 为点的位置设置对应的文字。
# 第一个参数是点的位置，第二个参数是点的文字提示。
plt.yticks([-2, -1.8, -1, 1.22, 3],
          [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$readly\ good$'])

# 设定XY轴的样式
# gca = 'get current axis'
ax = plt.gca()

# 隐藏坐标轴0,0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 图示
plt.legend(handles=[l1, l2], 
           labels = ['A1', 'B1'], 
           loc = 'best'
          )

plt.show()
