import matplotlib.pyplot as plt
import numpy as np


class Graph(object):
    def __init__(self):
        self.font = {
            'size': 13
        }

    # 设定图像输出:width,height
    plt.figure(figsize=(15, 7))

    # 子图的横向/纵向 间距
    plt.subplots_adjust(wspace=0.7, hspace=0.5)

    # 设定输出可为中文
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False

    # 折线图
    def broken_line(self):
        
        # 定义子图输出是 2行/3列 当前子索引为1，从左到右/从上到下排列
        a1 = plt.subplot(231)

        plt.title('双纵轴折线图', fontdict=self.font)

        # 输入数据:v1_data
        a1.plot(subjects, v1, label='v1')
        a1.set_ylabel('v1')
        a1.legend(loc='upper right', bbox_to_anchor=[-0.5, 0, 0.5, 1], fontsize=7)

        # 输入数据:v2_data 
        # 两种方式都可以，a1.twinx() 或者在生成一个subplot对象
        # a2 = a1.twinx() # 将ax1的x轴也分配给ax2使用
        a2 = plt.subplot(231)
        a2.plot(subjects, v2, 'r--', label='v2')
        # a2.set_ylabel('v2')
        a2.legend(loc='upper left', bbox_to_anchor=[1, 0, 0.5, 1], fontsize=7)

    def scatter(self):
        plt.subplot(232)
        plt.title('散点图', fontdict=self.font)
        x = range(50)
        y_jiangsu = [np.random.uniform(15, 25) for i in x]
        y_beijing = [np.random.uniform(5, 18) for i in x]
        plt.scatter(x, y_beijing, label='v1')
        plt.scatter(x, y_jiangsu, label='v2')
        plt.legend(loc='upper left', bbox_to_anchor=[1, 0, 0.5, 1], fontsize=7)

    def hist(self):
        plt.subplot(233)
        plt.title('直方图', fontdict=self.font)
        x = np.random.normal(size=100)
        plt.hist(x, bins=30)

    def bar_dj(self):
        plt.subplot(234)
        plt.title('堆积柱状图', fontdict=self.font)
        plt.bar(np.arange(len(v1)), v1, width=0.6, label='v1')
        for x, y in enumerate(v1):
            plt.text(x, y, y, va='top', ha='center')
        plt.bar(np.arange(len(v2)), v2, width=0.6, bottom=v1, label='v2')
        for x, y in enumerate(v2):
            plt.text(x, y + 60, y, va='bottom', ha='center')
        plt.ylim(0, 200)
        plt.legend(loc='upper left', bbox_to_anchor=[1, 0, 0.5, 1], fontsize=7)
        plt.xticks(np.arange(len(v1)), subjects)

    def bar_bl(self):
        plt.subplot(235)
        plt.title('并列柱状图', fontdict=self.font)
        plt.bar(np.arange(len(v1)), v1, width=0.4, label='v1')
        for x, y in enumerate(v1):
            plt.text(x - 0.2, y, y)
        plt.bar(np.arange(len(v2)) + 0.4, v2, width=0.4, label='v2')
        for x, y in enumerate(v2):
            plt.text(x + 0.2, y, y)
        plt.ylim(0, 110)
        plt.xticks(np.arange(len(v1)), subjects)
        plt.legend(loc='upper left', bbox_to_anchor=[1, 0, 0.5, 1], fontsize=7)

    def barh(self):
        plt.subplot(236)
        plt.title('水平柱状图', fontdict=self.font)
        plt.barh(np.arange(len(v1)), v1, height=0.4, label='v1')
        plt.barh(np.arange(len(v2)) + 0.4, v2, height=0.4, label='v2')
        plt.legend(loc='upper left', bbox_to_anchor=[1, 0, 0.5, 1], fontsize=7)
        plt.yticks(np.arange(len(v1)), subjects)


def main():
    g = Graph()
    g.broken_line()
    g.scatter()
    g.hist()
    g.bar_dj()
    g.bar_bl()
    g.barh()
    # plt.savefig('坐标轴类.png')
    plt.show()


if __name__ == '__main__':
    subjects = ['语文', '数学', '英语', '物理', '化学']
    v1 = [77, 92, 83, 74, 90]
    v2 = [63, 88, 99, 69, 66]
    main()

