import pickle

a_dict = {'da': 111, 2: [2, 3, 4], '23': {1: 2, 'd': 'dd'}}

file = open('data/pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()

with open('data/pickle_example.pickle','rb') as file:
    b_dict = pickle.load(file)
    print(b_dict)