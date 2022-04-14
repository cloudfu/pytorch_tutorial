from asyncio.windows_events import NULL
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
 
# 子图支持的数据数据类型和格式
class axis_type(Enum):
    broken_line  = 0

class broken_line_style:
    line_color = "red"
    line_width = 1.0
    line_stype = "--"

class matplat_util:

    def __init__(self,figsize,num=1):
        self.subplot_rows = 1
        self.subplot_cols = 1
        self.subplot_index = 1
        self.axis_types = 0
        plt.figure(num,figsize = figsize)

    # 设定子图的输出行、列、图标类型、当前图标索引
    def set_subplot(self,rows,cols,types):
        self.subplot_rows = rows
        self.subplot_cols = cols
        self.axis_types = types
        self.set_current_subplot(1)

    # 设定当前子图标索引
    def set_current_subplot(self,index):
        self.subplot_index = index
        plt.subplot(self.subplot_rows,self.subplot_cols,index)

    # 添加子图标数据集,axis_index:从0开始需要负责axis_types 索引，另外subplot 单独+1计算
    def add_dataset_axis(self,x,y,axis_index,clear_axis=False,color='red',linewidth=1.0, linestyle='--'):

        # 设定当前操作的子窗口
        self.set_current_subplot(axis_index + 1)

        # 是否清楚axis数据
        if clear_axis:
            plt.cla()

        # 折线图输出
        if self.axis_types[axis_index] == axis_type.broken_line:
            plt.plot(x,y,color,linewidth,linestyle)

    def show(self):
        plt.show()

    # def createTimer():
    #     t = threading.Timer(2, task)
    #     t.start()

    # count = 0
    # def task():
    #     global count
    #     global t
    #     count = count + 1
    #     if count > 5:
    #         print("cancel")
    #         return 
    #     print('Now:', time.strftime('%H:%M:%S',time.localtime()))
    #     createTimer()


    # createTimer()

x = np.linspace(-1, 1, 50)
y1 = 2*x + 1
y2 = x**2 + 1
axis_types = [axis_type.broken_line,axis_type.broken_line]

mlp = matplat_util(figsize=(8, 5))
mlp.set_subplot(2,2,axis_types)
mlp.add_dataset_axis(x,y1,0)
mlp.add_dataset_axis(x,y2,1)
mlp.show()
