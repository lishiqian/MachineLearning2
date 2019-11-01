import pandas as pd

df = pd.read_csv('data/stu_data.csv')
print(df)
print(df.loc[1, 'name'])

df.to_pickle('data/student.pickle')
 