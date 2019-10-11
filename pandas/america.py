import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
data1 = pd.read_csv('data/data_01.csv')
data2 = pd.read_csv('data/data_02.csv')
data3 = pd.read_csv('data/data_03.csv')
# 数据合并
data = pd.concat([data1, data2, data3])
# 数据预览
# print(data.head())
# print(data.info())
# print(data.describe())
# 缺失值处理
data['contbr_employer'].fillna('NOT PROVIDED', inplace=True)
data['contbr_occupation'].fillna('NOT PROVIDED', inplace=True)
# 数据转换
# print('共有{}位候选人，分别是'.format(len(data['cand_nm'].unique())))
# print(data['cand_nm'].unique())
parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
data['party'] = data['cand_nm'].map(parties)
# 查看两个党派的情况
data['party'].value_counts()
# 按照职业汇总对赞助总金额进行排序
a = data.groupby('contbr_occupation')['contb_receipt_amt'].sum().sort_values(ascending=False)[:20]
# print(a)
# 职业与雇主信息分析
occupation_map = {
  'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
  'INFORMATION REQUESTED':'NOT PROVIDED',
  'SELF' : 'SELF-EMPLOYED',
  'SELF EMPLOYED' : 'SELF-EMPLOYED',
  'C.E.O.':'CEO',
  'LAWYER':'ATTORNEY',
}
f = lambda x: occupation_map.get(x, x)
data.contbr_occupation = data.contbr_occupation.map(f)
emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
   'INFORMATION REQUESTED': 'NOT PROVIDED',
   'SELF': 'SELF-EMPLOYED',
   'SELF EMPLOYED': 'SELF-EMPLOYED',
}
f = lambda x: emp_mapping.get(x, x)
data.contbr_employer = data.contbr_employer.map(f)
# 数据筛选
# 赞助金额筛选
data = data[data['contb_receipt_amt'] > 0]
# 候选人筛选
data.groupby('cand_nm')['contb_receipt_amt'].sum().sort_values(ascending=False)
data_vs = data[data['cand_nm'].isin(['Obama, Barack', 'Romney, Mitt'])].copy()
# print(data_vs)
# 面元化数据
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(data_vs['contb_receipt_amt'], bins)
# print(labels)
# 分析党派和职业
by_occupation = data.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm.plot(kind='bar')


# 根据职业与雇主信息分组运算
def get_top_amounts(group, key, n=5):
    # 传入groupby分组后的对象，返回按照key字段汇总的排序前n的数据
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending=False)[:n]
grouped = data_vs.groupby('cand_nm')
s1 = grouped.apply(get_top_amounts, 'contbr_occupation', n=7)
s2 = grouped.apply(get_top_amounts, 'contbr_employer', n=10)
# 对赞助金额进行分组分析
grouped_bins = data_vs.groupby(['cand_nm', labels])
grouped_bins.size().unstack(0)
bucket_sums = grouped_bins['contb_receipt_amt'].sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
normed_sums[:-2].plot(kind='bar', stacked=True)


# 时间处理
data_vs['time'] = pd.to_datetime(data_vs['contb_receipt_dt'])
data_vs.set_index('time', inplace=True)
data_vs.head()
vs_time = data_vs.groupby('cand_nm').resample('M')['cand_nm'].count()
vs_time.unstack(0)
fig1, ax1 = plt.subplots(figsize=(32, 8))
vs_time.unstack(0).plot(kind='area', ax=ax1, alpha=0.6)
plt.show()
