import pandas as pd

def load_data():
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safaty', 'label']
    return pd.read_csv('data/car.csv', names=columns)


def convert2onehot(data):
    return pd.get_dummies(data)


data = load_data()
data = convert2onehot(data)
print(data.columns)
