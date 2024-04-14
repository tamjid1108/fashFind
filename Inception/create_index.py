import os
from visualFeatureExtract import returnVisualFeatures, Visualiser
from classifier import returnOneHot, returnTextWords, Classifier
import json
import pickle
import numpy as np
from annoy import AnnoyIndex
from preprocess_captions import get_query_vector2

training_data = pickle.load(
    open('training_data_checkpoints/training_data_17000.pkl', 'rb'))

vector_dim = len(next(iter(training_data.values())))
# vector_dim = 2346

distance_mode = 'angular'
learnt_data_space = "models/Inception/search_index.pkl"
training_dict_files = 'models/Inception/images_list.pkl'

t = AnnoyIndex(vector_dim, distance_mode)

for i, vector in enumerate(training_data.values()):
    t.add_item(i, vector)  # Add items with unique IDs

t.build(n_trees=10)
t.save(learnt_data_space)

images_list = list(training_data.keys())
with open(training_dict_files, 'wb') as file:
    pickle.dump(images_list, file)
