import json
import pickle

caption_file = "datasets/DeepFashion/captions.json"
colors_file = "models/Inception/color_list.pkl"

color_appended_captions_file = "datasets/DeepFashion/captions_color_appended.json"

with open(caption_file, 'r') as f:
    captions = json.load(f)

with open(colors_file, 'rb') as f:
    colors = pickle.load(f)

print(captions['WOMEN-Tees_Tanks-id_00007981-03_7_additional.jpg'])

for file in colors.keys():
    if file not in captions.keys():
        captions[file] = 'The colors are ' + ' '.join(colors[file])
    else:
        captions[file] += ' The colors are ' + ' '.join(colors[file])

print(captions['WOMEN-Tees_Tanks-id_00007981-03_7_additional.jpg'])

print(len(captions))

with open(color_appended_captions_file, 'w') as f:
    json.dump(captions, f)
