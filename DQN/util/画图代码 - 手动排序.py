import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from pylab import *
import csv
import os

def read_csv(filename):
    exampleFile = open(r'D:\学习\强化学习\资料\Deep Reinforcement Learning meets Graph Neural Networks exploring a routing optimization use case\impl\DRL-GNN\DQN\result\res/' + filename)  # 打开csv文件
    # exampleFile = open(r'D:\学习\强化学习\资料\Deep Reinforcement Learning meets Graph Neural Networks exploring a routing optimization use case\DRL-GNN-master\DQN/' + filename)  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    exampleData = list(exampleReader)  # csv数据转换为列表
    length_zu = len(exampleData)  # 得到数据行数

    mpl.rcParams['axes.unicode_minus'] = False
    x = list()
    y = list()

    for i in range(0, length_zu):  # 从第二行开始读取
        x.append(float(exampleData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
        y.append(float(exampleData[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表

    return x, y

def read_uti():
    # filename_list = os.listdir(r"C:\Users\惟今宵\Desktop\result\EMDQN\1/")
    filename_list = os.listdir(r"D:\学习\强化学习\资料\Deep Reinforcement Learning meets Graph Neural Networks exploring a routing optimization use case\impl\DRL-GNN\DQN\result\res/")
    print(filename_list)
    a = []

    for i in filename_list:
        x, y = read_csv(i)
        y1 = []
        for j in range(len(y)):
            y1.append(y[j])
        # y为reward列
        a.append(y1)

    return x, a, filename_list

def read_reward():
    # filename_list = os.listdir(r"C:\Users\惟今宵\Desktop\result\EMDQN\3/")
    filename_list = os.listdir(r"D:\学习\强化学习\资料\Deep Reinforcement Learning meets Graph Neural Networks exploring a routing optimization use case\impl\DRL-GNN\DQN\result\res/")
    print(filename_list)
    a = []
    for i in filename_list:
        x, y = read_csv(i)
        reward = 0
        y1 = []
        for j in range(len(y)):
            reward += y[j]
            if j % r_eps == r_eps - 1:
                # reward /= 10
                y1.append(reward)
                reward = 0
        # y为reward列
        a.append(y1)

    return x, a, filename_list

def filter(y):
    a = np.empty(eps)
    print(len(y))
    l=min(eps,len(y))
    # for i in range(len(y)):
    for i in range(l):
        a[i] = y[i]
    for i in range(l, eps):
        a[i] = 0
    return a

# eps和r数组
eps=200
r_eps = 10
x, y, filename_list= read_reward()

# 横纵坐标范围
x_tick = 0
y_tick = 160
x = range(0, eps)
# 平滑
for i in range(len(y)):
    y[i] = scipy.signal.savgol_filter(y[i], 10, 1)
    y[i] = filter(y[i])
    # y[i][0]=-20
    for j in range(len(y[i])):
        y[i][j] = y[i][j]

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 创建figure窗口
plt.figure(num=3, figsize=(8, 5))
# plt.title('throughput', size=15)
# 画曲线2
color = ['#00FF00','#DE3325','#0085FF','#FF9000','#9800FF','#F19EBB','#2B2B2B','#2FC0DB','#7591AE','#2B2B2B']
for i in range(len(filename_list)):
    filename_list[i] = filename_list[i].split('-')
    plt.plot(x, y[i], '-', color=color[i], label=filename_list[i][0], lw=1)
# y[5]+=33
# plt.plot(x, y[3], '-', color=color[3], label=filename_list[3][0], lw=1)
# plt.plot(x, y[1], '-', color=color[1], label=filename_list[1][0], lw=1)
# plt.plot(x, y[2], '-', color=color[2], label=filename_list[2][0], lw=1)
# plt.plot(x, y[5], '-', color=color[5], label=filename_list[5][0], lw=1)
# plt.plot(x, y[0], '-', color=color[0], label=filename_list[0][0], lw=1)
# plt.plot(x, y[4], '-', color=color[4], label=filename_list[4][0], lw=1)


plt.legend(fontsize=12)
# 设置坐标轴范围
plt.xlim((0, eps))
plt.ylim((x_tick, y_tick))

# 设置坐标轴名称，大小
plt.xlabel('iteration*10', size=18)
plt.ylabel('reward', size=18)

# 设置坐标轴刻度
my_x_ticks = np.arange(0, eps, 10)
my_y_ticks = np.arange(x_tick, y_tick, 10)
# 改变刻度
plt.xticks(my_x_ticks, size=18)
plt.yticks(my_y_ticks, size=18)

# 显示出所有设置
plt.grid()
plt.savefig("/chart")
plt.show()