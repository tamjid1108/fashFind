import os
from visualFeatureExtract import returnVisualFeatures, Visualiser
from classifier import returnOneHot, returnTextWords, Classifier
import json
import pickle
import numpy as np
from annoy import AnnoyIndex
from preprocess_captions import get_query_vector2


#### INPUT FILES ####

folder_path = "datasets/DeepFashion/images"
caption_file = "datasets/DeepFashion/captions.json"
text_embedding_file = "models/Inception/text_embedding_df.pkl"

shape_labels = "datasets/DeepFashion/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "datasets/DeepFashion/labels/texture/fabric_ann.txt"
pattern_texture_labels = "datasets/DeepFashion/labels/texture/pattern_ann.txt"


#### OUTPUT FILES ####
generic_text = "sleeveless print block pure color"
learnt_data_space = "models/Inception/search_index.pkl"
training_dict_files = 'models/Inception/images_list.pkl'


# 'angular' is suitable for cosine similarity. Can also try "euclidean", "manhattan", "hamming", or "dot"
distance_mode = 'angular'
classifier = Classifier(shape_labels_file=shape_labels,
                        fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
classifier.load()
visualiser = Visualiser()
visualiser.load()


file_list = os.listdir(folder_path)

folder_path_models = os.path.join(os.getcwd(), "models/Inception")

if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.makedirs(folder_path)

with open(caption_file, 'r') as f:
    captions = json.load(f)

with open(text_embedding_file, 'rb') as file:
    embeddings_model = pickle.load(file)

training_data = {}

kk = 0

for filename in file_list:
    filepath = os.path.join(folder_path, filename)
    tensor1 = np.array(returnVisualFeatures(visualiser, filepath))

    if filename in captions:
        text_vector = np.array(get_query_vector2(
            captions[filename], embeddings_model))
    else:
        text_vector = np.array(get_query_vector2(
            generic_text, embeddings_model))

    labels = np.array(get_query_vector2(
        " ".join(returnTextWords(classifier, filepath)), embeddings_model))

    one_hot_encoded_vectors = np.array(returnOneHot(classifier, filepath))

    concatenated_array = np.concatenate(
        (tensor1, text_vector, labels, one_hot_encoded_vectors), axis=0)
    training_data[filename] = concatenated_array
    kk += 1
    print("Training K = ", kk)
    # if kk % 100 == 0:
    #     print("Training K = ", kk)


vector_dim = len(next(iter(training_data.values())))
t = AnnoyIndex(vector_dim, distance_mode)

for i, vector in enumerate(training_data.values()):
    t.add_item(i, vector)  # Add items with unique IDs

t.build(n_trees=10)
t.save(learnt_data_space)

images_list = list(training_data.keys())
with open(training_dict_files, 'wb') as file:
    pickle.dump(images_list, file)
