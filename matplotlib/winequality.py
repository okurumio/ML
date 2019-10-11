import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option('precision', 3)
# 读取数据
df = pd.read_csv('data/winequality-red.csv', sep=';')
df.head(5)
# df.info()
df.describe()

# 全部属性
# 绘制箱线图
plt.style.use('ggplot')
colnm = df.columns.tolist()
fig = plt.figure(figsize=(10, 6))
for i in range(12):
    plt.subplot(2, 6, i+1)  # 创建单个子图
    sns.boxplot(df[colnm[i]], orient="v", width=0.5, color=color[2])  # 创建箱线图
    plt.ylabel(colnm[i], fontsize=12)  # 设置y轴坐标
plt.tight_layout()  # 自动调整子图参数
# 绘制单变量直方图
plt.figure(figsize=(10, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)  # 创建单个子图
    df[colnm[i]].hist(bins=100, color=color[3])
    plt.xlabel(colnm[i], fontsize=12)
    plt.ylabel('Frequency')
plt.tight_layout()
# plt.show()

# 酸度相关的特征
acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
               'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
plt.figure(figsize=(10, 4))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    v = np.log10(np.clip(df[acidityFeat[i]].values, a_min=0.001, a_max=None))
    plt.hist(v, bins=50, color=color[4])
    plt.xlabel('log(' + acidityFeat[i] + ')', fontsize=12)
    plt.ylabel('Frequency')
plt.tight_layout()

# 酸浓度直方图
plt.figure(figsize=(6, 3))
bins = 10**(np.linspace(-2, 2))
plt.hist(df['fixed acidity'], bins=bins, edgecolor='k', label='Fixed Acidity')
plt.hist(df['volatile acidity'], bins=bins, edgecolor='k', label='Volatile Acidity')
plt.hist(df['citric acid'], bins=bins, edgecolor='k', alpha=0.8, label='Citric Acid')
plt.xscale('log')
plt.xlabel('Acid Concentration (g/dm^3)')
plt.ylabel('Frequency')
plt.title('Histogram of Acid Concentration')
plt.legend()
plt.tight_layout()

# 总酸度
df['total acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.hist(df['total acid'], bins=50, color=color[5])
plt.xlabel('total acid')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(np.log(df['total acid']), bins=50, color=color[6])
plt.xlabel('log(total acid)')
plt.ylabel('Frequency')
plt.tight_layout()

# 甜度
df['sweetness'] = pd.cut(df['residual sugar'], bins=[0, 4, 12, 45], labels=["dry", "medium dry", "semi-sweet"])
plt.figure(figsize=(5, 3))
df['sweetness'].value_counts().plot(kind='bar', color=color[7])
plt.xticks(rotation=0)
plt.xlabel('sweetness', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()

# 红酒品质和理化特征的关系
# 箱线图
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.1)
colnm = df.columns.tolist()[:11] + ['total acid']
plt.figure(figsize=(10, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x='quality', y=colnm[i], data=df, color=color[1], width=0.6)
    plt.ylabel(colnm[i], fontsize=12)
plt.tight_layout()
# 热力图
sns.set_style("dark")
plt.figure(figsize=(10, 8))
colnm = df.columns.tolist()[:11] + ['total acid', 'quality']
mcorr = df[colnm].corr()
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.show()

# 密度和酒精浓度
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)
# plot figure
plt.figure(figsize=(6, 4))
sns.regplot(x='density', y='alcohol', data=df, scatter_kws={'s': 10}, color=color[1])
plt.xlim(0.989, 1.005)
plt.ylim(7, 16)

# 酸性物质含量和pH
acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide',
                   'sulphates', 'total acid']
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    sns.regplot(x='pH', y=acidity_related[i], data=df, scatter_kws={'s': 10}, color=color[1])
plt.tight_layout()
print("Figure 10: pH vs acid")

# 酒精浓度，挥发性酸和品质的关系
plt.style.use('ggplot')
# 回归图
sns.lmplot(x='alcohol', y='volatile acidity', hue='quality', data=df, fit_reg=False, scatter_kws={'s': 10}, size=5)
# 回归图
sns.lmplot(x='alcohol', y='volatile acidity', col='quality', hue='quality', data=df, fit_reg=False, size=3, aspect=0.9,
           col_wrap=3, scatter_kws={'s': 20})

# pH，非挥发性酸，和柠檬酸
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)
plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], c=df['pH'], vmin=2.6, vmax=4, s=15, cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation=0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4, 18)
plt.ylim(0, 1)
