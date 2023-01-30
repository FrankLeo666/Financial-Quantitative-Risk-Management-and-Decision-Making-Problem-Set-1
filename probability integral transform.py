import numpy as np
import scipy.stats as st
import seaborn as sns

np.random.seed(0)

# 产生正态分布随机数 generate normal distributed random numbers
x1 = np.random.normal(0, 1, size=1000000)

# 产生指数分布随机数 generate exponential distributed random numbers
x2 = np.random.exponential(1, size=1000000)

# 产生均匀分布随机数 generate uniformly distributed random numbers
u = np.random.uniform(0, 1, size=10000)

# 累积函数的逆
q1 = st.norm(0, 1).ppf(u)   # ppf is the inverse function of cdf
q2 = st.expon(1).ppf(u)

# 画图 plot
h1 = sns.jointplot(u, q1)
h1.set_axis_labels("original", "transformed", fontsize=10)
h1.savefig("probability integral transform of normal distribution.png")

h2 = sns.jointplot(u, q2)
h2.set_axis_labels("original", "transformed", fontsize=10)
h2.savefig("probability integral transform of exponential distribution.png")
