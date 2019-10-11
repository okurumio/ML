import pandas as pd
import numpy as np
from sqlalchemy import create_engine


class pandas:
    def __init__(self):
        self.df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6],
                           'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    # io流
    def ioStrean(self):
        pd.read_csv("a.csv", sep=',')  # CSV & 文本文件
        pd.read_json("a.json", encoding='utf-8')  # json文件
        pd.read_excel('lemon.xlsx', sheet_name='student')  # excel文件
        # 操作数据库
        sql = ''
        engine = create_engine('mysql+pymysql://root:password@localhost:3306/databasename?charset=utf8')
        pd.read_sql_query(sql, engine)

    # 索引
    def select(self):
        s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
        s1.loc()  # 按标签索引
        s1.iloc()  # 按位置索引
        s1.reindex()  # 重新索引(找不到的标为NaN)
        s1.sample()  # 从对象轴返回随机的项目样本。
        s1.at()  # 只访问标量值
        s1.iat()  # 只访问标量值
        s1.isin('')  # 检查系列中是否包含值,返回布尔值
        s1.where()  # 返回相同形状的对象,条件外的值为NaN
        s1.mask()  # 返回相同形状的对象,条件内的值为NaN
        s1.query()  # 使用布尔表达式查询DataFrame的列
        s1.duplicated()  # 返回一个布尔向量，其长度为行数，表示行是否重复
        s1.drop_duplicates()  # 删除重复的行
        s1.get('')  # 返回默认值
        s1.lookup()  # 提取一组值,可以返回numpy数组
        pd.Index()  # 索引对象
        pd.MultiIndex.from_arrays('')  # 分层索引
        s1.get_level_values()  # 方法将返回特定级别每个位置的标签向量
        s1.sort_index()  # 对索引排序
        s1.take('')  # 沿轴返回给定位置索引中的元素

    # 合并,重塑,透视
    def concat(self):
        # 合并
        s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
        pd.concat(s1, s1)  # 沿轴执行连接操作
        s1.append(s1)  # 链接
        pd.Series('')  # 连接混合Series和DataFrame对象
        pd.merge_ordered()  # 为时序数据等有序数据设计的可选填充/插值执行合并
        pd.merge_asof()  # 执行asof合并
        # 重塑/透视
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6],
                           'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        # index(用于制作新帧索引的列),columns(用于制作新框架列的列),values(用于填充新框架值的列)
        df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        df.stack()  # 将指定级别从列堆叠到索引
        df.unstack()
        # 将DataFrame从宽格式展开为长格式，可选择设置标识符变量,id_vars(用作标识符变量的列),value_vars(要拆开的列)
        df.melt(id_vars='foo', value_vars='bar')
        # 创建电子表格式数据透视表
        pd.pivot_table(df, values='zoo', index='bar', columns='foo', margins=True)
        # 计算两个（或更多）因子的简单交叉列表
        pd.crosstab(df, df, normalize='columns')
        # 将数据值分段并排序到箱中,bin值为离散间隔
        pd.cut(df, bins=3)
        # 将分类变量转换为虚拟/指示变量
        pd.get_dummies(s1)
        # 将对象编码为枚举类型或分类变量
        pd.factorize()
        # 将列表类的每个元素转换为行，复制索引值
        s1.explode()

    def str(self):
        s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
        s.str.lower()  # 转换小写
        s.str.upper()  # 转换大写
        s.str.len()  # 输出长度
        s.str.cat()  # 连接字符串
        s.str.extract()  # 提取

    def deficiency(self):
        s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
        pd.isna(s)  # 检测类似数组的对象的缺失值,空返回true
        pd.notna(s)  # 检测类似数组的对象的缺失值,不空返回true
        s.fillna()  # 使用指定的方法填充NA / NaN值
        s.dropna()  # 删除缺失值
        s.interpolate()  # 根据不同的方法插值
        s.replace()  # 替换给定的值

        s.astype('category')  # 将对象类型替换为Categorize,提升速度，节约空间

    def computation(self):
        s = pd.Series(np.random.randn(8))
        s.pct_change()  # 计算给定数量的周期内的百分比变化
        s.cov(s)  # 可用于计算序列之间的协方差
        self.df.corr(method="pearson")  # 计算相关性
        # method参数:pearson (default)皮尔逊相关系数,kendall肯德尔等级相关系数,spearman斯皮尔曼的等级相关系数
        self.df.corrwith(s)  # 计算不同对象相关性
        self.df.rank()  # 生成数据排名
        # window函数
        self.df.rolling(window=1).mean()  # 指定周期求平均值
        self.df.expanding(min_periods=5).sum()  # 从第五项开始累计求和
        self.df.ewm(com=5.0).mean()  # 计算移动平均

    def group(self):
        self.df.groupby(['foo', 'bar'])

    def time(self):
        pd.to_datetime('12-11-2010 00:00', format='%d-%m-%Y %H:%M')  # format(加快转换速度)
        pd.to_datetime([1, 2, 3], unit='D', origin=pd.Timestamp.now())  # 指定开始日期
        pd.date_range()  # 返回固定频率的DatetimeIndex,默认频率为日历日
        pd.bdate_range()  # 返回固定频率的DatetimeIndex,默认频率为工作日


# set_option()
# True就是可以换行显示。设置成False的时候不允许换行
pd.set_option('expand_frame_repr', False)
# 显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列。如果比较多又不允许换行，就会显得很乱。
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# 显示小数点后的位数
pd.set_option('precision', 5)
# truncate表示截断，info表示查看信息，一般选truncate
# pd.set_option('large_repr', A)
# 列长度
pd.set_option('max_colwidth', 5)
# 绝对值小于0.5的显示0.0
pd.set_option('chop_threshold', 0.5)
# 显示居中还是左边
pd.set_option('colheader_justify', 'left')
# 横向最多显示多少个字符， 一般80不适合横向的屏幕，平时多用200.
pd.set_option('display.width', 200)



