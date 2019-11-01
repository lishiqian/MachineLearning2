import pandas as pd
import numpy as np

dates = pd.date_range("20191031", periods=6)
df = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
print(df)
'''
             A     B     C   D
2019-10-31   0   NaN   2.0   3
2019-11-01   4   5.0   NaN   7
2019-11-02   8   9.0  10.0  11
2019-11-03  12  13.0  14.0  15
2019-11-04  16  17.0  18.0  19
2019-11-05  20  21.0  22.0  23
'''

# 将行中有null中的值都掉
print(df.dropna(axis=0, how='any'))   # how = ['any', 'all'] all 全部是NaN才丢掉
'''
             A     B     C   D
2019-11-02   8   9.0  10.0  11
2019-11-03  12  13.0  14.0  15
2019-11-04  16  17.0  18.0  19
2019-11-05  20  21.0  22.0  23
'''

# 将null填上值
print(df.fillna(value=0))
'''
             A     B     C   D
2019-10-31   0   0.0   2.0   3
2019-11-01   4   5.0   0.0   7
2019-11-02   8   9.0  10.0  11
2019-11-03  12  13.0  14.0  15
2019-11-04  16  17.0  18.0  19
2019-11-05  20  21.0  22.0  23
'''

# 判读值是不是null
print(df.isnull())
'''
                A      B      C      D
2019-10-31  False   True  False  False
2019-11-01  False  False   True  False
2019-11-02  False  False  False  False
2019-11-03  False  False  False  False
2019-11-04  False  False  False  False
2019-11-05  False  False  False  False
'''

# 判读有没有null值
print(np.any(df.isnull()) == True)
# True