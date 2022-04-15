from asyncio.windows_events import NULL
import math
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
 
# 子图支持的数据数据类型和格式
class chart_type(Enum):
    line  = 0
    scatter = 1
    image = 2


# 子图的样式类型
class axis_style:

    def __init__(self,chart_type,title="defalut",animator=False):
        self.__chart_type = chart_type
        self.__title = title
        self.__animator = animator

    def get_chart_type(self):
        return self.__chart_type

    def get_chart_title(self):
        return self.__title

    def get_animator(self):
        return self.__animator

# 颜色样式
class chart_styles:    
    color_names = {
        'aliceblue':            '#F0F8FF',
        'antiquewhite':         '#FAEBD7',
        'aqua':                 '#00FFFF',
        'aquamarine':           '#7FFFD4',
        'azure':                '#F0FFFF',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'black':                '#000000',
        'blanchedalmond':       '#FFEBCD',
        'blue':                 '#0000FF',
        'blueviolet':           '#8A2BE2',
        'brown':                '#A52A2A',
        'burlywood':            '#DEB887',
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',
        'coral':                '#FF7F50',
        'cornflowerblue':       '#6495ED',
        'cornsilk':             '#FFF8DC',
        'crimson':              '#DC143C',
        'cyan':                 '#00FFFF',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'floralwhite':          '#FFFAF0',
        'forestgreen':          '#228B22',
        'fuchsia':              '#FF00FF',
        'gainsboro':            '#DCDCDC',
        'ghostwhite':           '#F8F8FF',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'indigo':               '#4B0082',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'lime':                 '#00FF00',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navajowhite':          '#FFDEAD',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olive':                '#808000',
        'olivedrab':            '#6B8E23',
        'orange':               '#FFA500',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'silver':               '#C0C0C0',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'teal':                 '#008080',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'wheat':                '#F5DEB3',
        'white':                '#FFFFFF',
        'whitesmoke':           '#F5F5F5',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'}

    # 线段样式
    solid_line = '-'       
    dashed_line = '--'      
    dash_dot_line = '-.'      
    dotted_line = ':'   
    

# 数据输出格式和样式
class data_chart_style:

    def __init__(self,line_color=chart_styles.color_names["red"],line_width=1.0,line_stype=chart_styles.dashed_line):
        self.line_color = line_color
        self.line_width = line_width
        self.line_stype = line_stype


class matplat_util:

    def __init__(self,figsize,num=1):
        self.subplot_rows = 1
        self.subplot_cols = 1
        self.subplot_index = 1
        self.axis_styles = 0

        self.font = {'size': 11}
        # 设定输出可为中文
        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False

        # 子图的横向/纵向 间距
        plt.subplots_adjust(wspace=0.7, hspace=0.5)

        plt.figure(num,figsize = figsize)

    # 设定子图的输出行、列、图标类型、当前图标索引
    def create_subplot(self,rows,cols,style):
        self.subplot_rows = rows
        self.subplot_cols = cols
        self.axis_styles = style
        self.set_current_subplot(1)

    # 设定当前子图标索引
    def set_current_subplot(self,index):
        self.subplot_index = index
        return plt.subplot(self.subplot_rows,self.subplot_cols,index)

    # 添加子图标数据集,axis_index:从0开始需要负责axis_types 索引，另外subplot 单独+1计算
    def add_axis_dataset(self,axis_index,data_label="",x_data="",y_data="",img_data=""):

        # 设定当前操作的子窗口
        current_axis = self.set_current_subplot(axis_index + 1)

        # axis标题
        plt.title(self.axis_styles[axis_index].get_chart_title(), fontdict=self.font)

        # 是否清楚axis数据
        # if clear_axis:

        # plt.ion()

        # 折线图输出
        if(self.axis_styles[axis_index].get_chart_type() == chart_type.line):
            self.__draw_line(current_axis,axis_index,data_label,x_data,y_data)
            current_axis.legend(loc='best')
        # 散点输出
        elif (self.axis_styles[axis_index].get_chart_type() == chart_type.scatter):
            self.__draw_scatter(current_axis,axis_index,data_label,x_data,y_data)
            current_axis.legend(loc='best')
        # 图像输出
        elif (self.axis_styles[axis_index].get_chart_type() == chart_type.image):
            self.__draw_image(current_axis,img_data)
            # current_axis.legend(loc='best')

        # plt.ioff()

        # 图例
        
    # 绘制图片
    def __draw_image(self,axis,image):
        axis.imshow(image)


    # 绘制折线
    def __draw_line(self,axis,axis_index,label_name,x,y):

        if(self.axis_styles[axis_index].get_animator()):
            for i in range(len(x)):
                axis.cla()
                axis.plot(x[0:i],y[0:i],label=label_name)
                plt.pause(0.1)
        else:
            axis.plot(x,y,label=label_name)
            
    # 绘制散点图
    def __draw_scatter(self,axis,axis_index,label_name,x,y):
        if(self.axis_styles[axis_index].get_animator()):
            for i in range(len(x)):
                axis.cla()
                axis.scatter(x[0:i],y[0:i],label=label_name)
                plt.pause(0.1)
        else:
            axis.scatter(x,y,label=label_name)


    # 开启/关闭数据交互模式
    def set_interactive(status):
        if(status):
            plt.ion()
        else:
            plt.ioff()


    def show(self):
        plt.show()


# def createTimer(x,y):
#     t = threading.Timer(1, task(x,y))
#     t.start()

# count = 0
# def task(x,y):
#     global count
#     global t
#     count = count + 1
#     print(count)
#     if count >= len(x):
#         print("cancel")
#         return 
#     mlp.add_axis_dataset("lable1",x[0:count],y[0:count],0)
#     createTimer(x,y)

# createTimer(x,y1)

# 数据展示
x = np.linspace(-1, 1, 50)
y1 = 2 * x + 1
y2 = x ** 2 + 1
y3 = np.tan(x)
y4 = np.cos(x)

# 图像展示
image_blue = np.zeros((400,400,3),dtype=np.uint8)
image_blue[:]=(0,0,255)

image_red = np.zeros((400,400,3),dtype=np.uint8)
image_red[:]=(0,255,0)

image_green = np.zeros((400,400,3),dtype=np.uint8)
image_green[:]=(255,0,0)

axis_styles = [axis_style(chart_type.line,"折线图",False),
               axis_style(chart_type.scatter,"散点图",False),
               axis_style(chart_type.scatter,"折线图",False),
               axis_style(chart_type.image,"蓝色",False),
               axis_style(chart_type.image,"红色",False),
               axis_style(chart_type.image,"绿色",False)]

mlp = matplat_util(figsize=(8, 5))
mlp.create_subplot(2,3,axis_styles)
mlp.add_axis_dataset(0,data_label="line",x_data=x,y_data=y1)
mlp.add_axis_dataset(1,data_label="scatter",x_data=x,y_data=y2)
mlp.add_axis_dataset(2,data_label="scatter",x_data=x,y_data=y3)

mlp.add_axis_dataset(3,data_label="blue",img_data=image_blue)
mlp.add_axis_dataset(4,data_label="red",img_data=image_red)
mlp.add_axis_dataset(5,data_label="green",img_data=image_green)

mlp.show()





