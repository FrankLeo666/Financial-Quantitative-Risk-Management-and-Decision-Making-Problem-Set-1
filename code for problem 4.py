import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# 读取股价历史数据
os.chdir("C:\\Users\\liufengqi\\Desktop")
df = pd.read_table("股价历史数据.txt", sep="\t", header=0, encoding="UTF-8", index_col=0)
data = np.asarray(df)
stock1 = data[:, 0]
stock2 = data[:, 1]
stock3 = data[:, 2]

# 计算日收益率均值、协方差矩阵及逆矩阵
E = np.asarray(df.mean(axis=0))
cov = np.asarray(df.cov())
inv_cov = np.linalg.inv(cov)

# 求协方差矩阵的特征值和特征向量
eigvals, eigvecs = np.linalg.eig(cov)
# print('特征值数组:\n', eigvals)
# print('特征向量:\n', eigvecs)
ld = np.asarray([[eigvals[0] ** 0.5, 0, 0], [0, eigvals[1] ** 0.5, 0], [0, 0, eigvals[2] ** 0.5]])


# 求所有样本点距离均值点的马氏距离的均值 q
q = sum([np.sqrt((x - E).T @ inv_cov @ (x - E)) for x in data]) / df.shape[0]
# print(q)

# 求出有多少样本点会被包含在 Ellipsoid 之内
count = 0
for i in data:
    dis = np.sqrt((i - E).T @ inv_cov @ (i - E))
    if dis < q:
        count += 1

per = count / df.shape[0]
print("The percentage of the points covered by the ball is %.2f" % (per * 100) + "%")

# 下面开始画 Ellipsoid
def DrawEllipsoid(radius, MeanValue):
    # 先画一个单位球
    # center and radius
    center = MeanValue
    radius = radius  # 可以把 q 改为 (1+5%)q、(1+10%)q

    # data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ball = np.asarray([x, y, z])

    # 通过矩阵乘法（线性变换）将单位球拉伸、旋转为 Ellipsoid
    ellpsoid = []
    for i in range(0, 100):
        ellpsoid.append(np.dot(np.dot(eigvecs, ld), ball[:, :, i]))

    ellpsoid = np.asarray(ellpsoid)
    x = ellpsoid[:, 0, :] + center[0]
    y = ellpsoid[:, 1, :] + center[1]
    z = ellpsoid[:, 2, :] + center[2]
    return x, y, z

# 开始绘图
# plot
fig = plt.figure(figsize=(100, 100))

# wire frame
x1, y1, z1 = DrawEllipsoid(q, E.tolist())
x2, y2, z2 = DrawEllipsoid(1.05 * q, E.tolist())
x3, y3, z3 = DrawEllipsoid(1.1 * q, E.tolist())
x4, y4, z4 = DrawEllipsoid(0.95 * q, E.tolist())
x5, y5, z5 = DrawEllipsoid(0.9 * q, E.tolist())

print(x1.shape)

ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(x1, y1, z1, rstride=10, cstride=10, color="blue", linewidth=0.5, label = "q")
# ax.plot_wireframe(x2, y2, z2, rstride=10, cstride=10, color="green", linewidth=0.5, label="(1+5%)q")
# ax.plot_wireframe(x3, y3, z3, rstride=10, cstride=10, color="purple", linewidth=0.5, label="(1+10%)q")
# ax.plot_wireframe(x4, y4, z4, rstride=10, cstride=10, color="yellow", linewidth=0.5, label="(1-5%)q")
# ax.plot_wireframe(x5, y5, z5, rstride=10, cstride=10, color="black", linewidth=0.5, label="(1-10%)q")

# 把样本点也画出来
ax.scatter(stock1, stock2, stock3, s=1, c="red", zorder=1)

ax.set_zlabel('002202.SZ', fontdict={'size': 10, 'color': 'blue'})
ax.set_ylabel('603885.SH', fontdict={'size': 10, 'color': 'blue'})
ax.set_xlabel('600048.SH', fontdict={'size': 10, 'color': 'blue'})
plt.title("q = %f" % q + "\n" + "The percentage of the points covered by the ball is %.2f" % (per * 100) + "%",
          fontsize=10)
plt.legend(loc=1, ncol=1)

# # 以下为最后一小题的代码，请勿与前面的代码同时运行
#
# # 读取2021年股价数据
# df1 = pd.read_table("2021股价数据.txt", sep="\t", header=0, encoding="UTF-8", index_col=0)
# data1 = np.asarray(df1)
# newstock1 = data1[:, 0]
# newstock2 = data1[:, 1]
# newstock3 = data1[:, 2]
#
# # 绘制2021股价散点图
#
# ax.scatter(newstock1, newstock2, newstock3, s=3, c="green", zorder=1, label = "Data of 2021")
#
# # 历史数据的散点也绘制出来，以供对比
# ax.scatter(stock1, stock2, stock3, s=3, c="red", zorder=1, label = "Data of 2018-2020")
#
# ax.set_zlabel('002202.SZ', fontdict={'size': 10, 'color': 'blue'})
# ax.set_ylabel('603885.SH', fontdict={'size': 10, 'color': 'blue'})
# ax.set_xlabel('600048.SH', fontdict={'size': 10, 'color': 'blue'})
#
# # 计算新数据有多少被包含在 Ellipsoid 之内
# count = 0
# for i in data1:
#     dis = np.sqrt((i - E).T @ inv_cov @ (i - E))
#     if dis < q:
#         count += 1
# print(count)
# per = count / df1.shape[0]
# print(per)
# print("The percentage of the points covered by the ball is %.2f" % (per * 100) + "%")
# plt.title("q = %f" % q + "\n" + "The percentage of the new points covered by the ball is %.2f" % (per * 100) + "%",
#           fontsize=10)
# plt.legend(loc=1, ncol=1)
#
plt.show()
