import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# 将 iris 数组集的第一个列的数据范围缩放为 0 到 1
Smax, Smin = iris.max(), iris.min()
S = (iris - Smin)/iris.ptp()
# print(S)

# 根据百分比大小返回元素
a = np.percentile(iris, q=[5, 95])
# print(a)

# 找出数组的缺失值
iris2 = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
iris2[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# print("Number of missing values: \n", np.isnan(iris2[:, 0]).sum())
# print("Position of missing values: \n", np.where(np.isnan(iris2[:, 0])))

# 返回数组中出现最多的元素
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])

# 找出数组中某元素满足第一次大于某数的下标
b = np.argwhere(iris[:, 3].astype(float) > 1.0)[0]
print(b)

# 去掉所有缺失值
iris2[~np.isnan(iris2)]
