import pickle

dicts = []

files = ['color_dict_0.pkl', 'color_dict_10000.pkl',
         'color_dict_20000.pkl', 'color_dict_30000.pkl']


for file in files:
    with open(file, 'rb') as f:
        dicts.append(pickle.load(f))

for d in dicts[1:]:
    dicts[0].update(d)


with open('models/Inception/color_list.pkl', 'wb') as f:
    pickle.dump(dicts[0], f)
