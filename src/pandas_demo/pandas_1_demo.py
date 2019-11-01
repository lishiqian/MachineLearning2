import pandas as pd
import numpy as np

s = pd.Series([1, 3, 6, np.NaN, 44, 15])
print(s)

# 日期数据
dates = pd.date_range("20191031", periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
print('df.dtypes:', df.dtypes)  # 打印每一列的数据类型
print('df.index:', df.index)  # 打印每一列序号
print('df.columns：', df.columns)  # 打印列明
print('df.values:', df.values)  # 打印值

print('df.describe():', df.describe())  # 计算 count mean 方差 最大值等基本信息

# 转置
print(df.T)

# 排序
print(df.sort_index(axis=1, ascending=False))
print(df.sort_index(axis=1, ascending=True))
