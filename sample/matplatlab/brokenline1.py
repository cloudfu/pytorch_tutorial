import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot([-2, 2, 3, 4], [-10, 20, 25, 5])
ax.spines['top'].set_visible(False)     #顶边界不可见
ax.xaxis.set_ticks_position('bottom')  # ticks 的位置为下方，分上下的。
ax.spines['right'].set_visible(False)   #右边界不可见
ax.yaxis.set_ticks_position('left')  

# "outward"
# 移动左、下边界离 Axes 10 个距离
#ax.spines['bottom'].set_position(('outward', 10))
#ax.spines['left'].set_position(('outward', 10))

# "data"
# 移动左、下边界到 (0, 0) 处相交
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# "axes"
# 移动边界，按 Axes 的百分比位置
#ax.spines['bottom'].set_position(('axes', 0.75))
#ax.spines['left'].set_position(('axes', 0.3))

plt.show()