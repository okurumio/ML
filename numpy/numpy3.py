import numpy as np


params = dict(
    fname="data.csv",
    delimiter=',',
    usecols=(4, 5, 6, 7),
    unpack=True
)
highPrice, lowPrice, endPrice, turnover = np.loadtxt(**params)
print("加权平均数：", np.average(endPrice, weights=turnover))
print("最高价最大值：", highPrice.max())
print("最高价最小值", lowPrice.min())
print("最高价极差：", highPrice.ptp())
print("最低价极差：", lowPrice.ptp())
print("中位数：", np.median(endPrice))
print("方差：", endPrice.var())
logReturns = np.diff(np.log(endPrice))  # 对数收益率
annual_volatility = logReturns.std()/logReturns.mean()*np.sqrt(252)
monthly_volatility = logReturns.std()/logReturns.mean()*np.sqrt(12)
print("年波动率", annual_volatility)
print("月波动率", monthly_volatility)
