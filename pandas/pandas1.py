import pandas as pd
import numpy as np


a = pd.Series([1, 2, 3, np.nan, 4])
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
b = pd.date_range('20130101', periods=6)
c = pd.DataFrame(np.random.randn(6, 4), index=b, columns=list('ABCD'))
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
# print(df)
# print(pd.concat(pieces))

df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'baz': [1, 2, 3, 4, 5, 6],
                       'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
d = df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
e = df.stack()
f = df.melt(id_vars='foo', value_vars=['bar', 'baz'])
# pd.pivot_table(df, values='zoo', index='bar', columns='foo')
pd.cut(a, bins=3)
pd.factorize(a)
s.str.len()
# df.corr(method="spearman")
# df.expanding(min_periods=5).sum()
# a = df.groupby(['foo', 'bar'])
print(pd.Timestamp.now())
