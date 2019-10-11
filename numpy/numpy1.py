import numpy as np


# 属性
def numpy_attribute():
    array = np.arange(15).reshape(3, 5)
    print(array.ndim)  # 数组的轴数
    print(array.shape)  # 查看大小形状
    print(array.size)  # 数组的元素总数
    print(array.dtype)  # 描述数组中元素类型的对象
    print(array.itemsize)  # 数组中每个元素的大小
    print(array.flags)  # 有关阵列内存布局的信息
    print(array.strides)  # 遍历数组时每个维度中的字节元组
    print(array.data)  # Python缓冲区对象指向数组的数据的开头
    print(array.itemsize)  # 获得一个数组元素的长度，以字节为单位
    print(array.nbytes)  # 数组元素消耗的总字节数。

    print(array.T)  # 转置数组。
    print(array.real)  # 数组的真实部分。
    print(array.imag)  # 数组的虚部。
    print(array.flat)  # 数组上的一维迭代器。
    print(array.ctypes)  # 一个简化数组与ctypes模块交互的对象。


# 创建ndarray
def numpy_initialize():
    arr = np.array([1, 2, 3])
    ndarr = np.array([i for i in range(10)])  # numpy创建数组
    zeroarr = np.zeros(shape=(3, 4), dtype='float')  # 创建一个全0的数组，shape为形状
    onearr = np.ones(shape=(2, 2))  # 创建一个全1的数组
    emptyarr = np.empty(shape=(1, 2))  # 创建一个随机数组
    fularr = np.full(shape=(3, 3), fill_value=8)  # 创建一个数组, fill_value指定填充物
    arange = np.arange(0, 20, step=2)  # 起点，终点，步长
    linarr = np.linspace(start=0, stop=10, num=4)  # 开始，结束，个数
    # random:
    randintarr = np.random.randint(low=0, high=100)  # 产生随机数low-high
    np.random.seed(6)  # 如果我们每次想产生想同的随机数只需要种下一样的种子就可以了
    randomarr = np.random.random()  # 产生0-1的随机数
    normalarr = np.random.normal(loc=0.0, scale=1.0)  # 产生服从高斯分布的随机数,loc是均值，scale是方差
    x, y = np.ogrid[0:10:6j, 0:10:4j]  # 生成二维数组，第一个数组是以纵向产生的，即数组第二维的大小始终为1。第二个数组是以横向产生的，即数组第一维的大小始终为1
    print(x, np.shape(x))
    print(y,np.shape(y))


# 通用函数
def numpy_ufunc():
    X = np.arange(15).reshape(3, 5)
    Y = np.arange(46).reshape(3, 5)
    print(X/2)  # 除法
    print(X//2)  # 取整除法
    print(X ** 2)  # 幂运算
    print(X % 2)  # 取余
    np.log(X)
    np.log2(X)
    np.log10(X)
    np.abs(X)  # 计算数组各元素的绝对值
    np.fabs(X)  # 计算数组各元素的绝对值
    np.sqrt(X)  # 计算数组各元素的平方根
    np.square(X)  # 计算数组各元素的平方
    np.exp(X)  # 计算数组各元素的指数值
    np.around(X, decimals=1)  # 函数返回指定数字的四舍五入值,decimals舍入的小数位数
    np.floor(X)  # 返回数字的下舍整数
    np.ceil(X)  # 返回数字的上入整数。
    np.reciprocal(X)  # 函数返回参数逐元素的倒数
    np.power(X, 2)  # 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
    np.mod(X, Y)  # 计算输入数组中相应元素的相除后的余数
    np.sum(X)  # 求和
    np.min(X)  # 最小值
    np.max(X)  # 最大值
    np.prod(X)  # 连乘
    np.mean(X)  # 求均值
    np.median(X)  # 中位数
    np.percentile(X, q=50)  # 百分位点
    np.var(X)  # 方差
    np.std(X)  # 标准差
    np.sum(X <= 3)  # 满足条件的求和
    np.sum((X > 3) & (X > 10))  # 并且
    np.sum((X % 2 == 0) | (X > 10))  # 或者，因为左右两边都是数组（bool类型的），所以这里用位运算
    np.sum(~(X == 0))  # 非
    np.count_nonzero(X < 3)  # 满足条件记录不是0的个数
    np.any(X < 0)  # any是否有一个，有一个就满足。
    np.all(X >= 0)  # all所有的元素都要满足
    np.ptp()  # 求最大值与最小值的差
    np.bincount(X)  # 计算非负int数组中每个值的出现次数
    np.ceil(X)  # 以元素方式返回输入的上限
    np.conj(X)  # 以元素方式返回复共轭
    np.cov()
    np.argmin(X)
    np.argmax(X)
    np.linalg.inv(X)  # 逆矩阵


def numpy_operation():
    X = np.arange(15).reshape(3, 5)
    # 排序
    np.sort(X)
    np.argsort(X)

    X.reshape(3, 5)  # 改变形状
    np.resize(X, (2, 3))  # 返回具有指定形状的新数组
    np.ravel(X)  # 返回一个连续的扁平数组

    # 堆叠
    np.concatenate([X, X], axis=0)  # 拼接数组，axis是维度
    np.vstack([X, X])  # 在垂直方向进行堆叠
    np.hstack([X, X])  # 在垂直方向进行堆叠
    np.column_stack((X, X))  # 将1-D阵列作为列堆叠成2-D阵列。

    # 分割
    np.split()
    np.vsplit()
    np.hsplit()


# 设置输出样式
np.set_printoptions(threshod='nan')  # 打印整个数组
np.set_printoptions(precision = 4)  # 设置浮点精度
np.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan',precision=8,suppress=False, threshold=1000, formatter=None)  # 返回默认选项
np.set_printoptions(threshold=5)  # 概略显示

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
# np.intersect1d(a, b)  # 返回公共元素
# np.setdiff1d(a, b)  # 从a中删除b
# np.where(a, b)  # 返回相同元素的下
# np.vectorize(Universal_Function)  # 将函数向量化
# np.unique()  # 去重并排序输出
# np.clip(a, a_min=10, a_max=30)  # 设定数组元素的上下限
