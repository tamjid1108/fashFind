import os
from .visualFeatureExtract import returnVisualFeatures, Visualiser
from .classifier import returnOneHot, returnTextWords, Classifier
import pickle
import numpy as np
from annoy import AnnoyIndex
# from image_viewer import  mv3
from .preprocess_captions import get_query_vector2
from .metric import ang_avg

# Replace with folder you trained over, so that similar images can be retrieved
train_folder = "datasets/DeepFashion/images"
number_retrieved = 6

#### INPUT FILE NAMES ####
# query_image = "/Users/gsp/Downloads/images/MEN-Tees_Tanks-id_00000390-13_1_front.jpg"
# query_image = "/Users/gsp/Documents/photus/meee.jpeg"
# query_image = "/Users/gsp/Downloads/1516264345931.jpeg"
# query_text = "woman floral" ## Make sure empty code handled

learnt_data_space = "models/Inception/search_index_color.pkl"
training_dict_files = 'models/Inception/images_list_color.pkl'
# 'angular' is suitable for cosine similarity. Can also try "euclidean", "manhattan", "hamming", or "dot"
distance_mode = 'angular'
vector_length = 2346
text_embeddings_file = 'models/Inception/text_embedding_df.pkl'
text_weight = 2000

shape_labels = "datasets/DeepFashion/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "datasets/DeepFashion/labels/texture/fabric_ann.txt"
pattern_texture_labels = "datasets/DeepFashion/labels/texture/pattern_ann.txt"


def Nearest_images(query_image, query_text, w=text_weight):
    classifier = Classifier(shape_labels_file=shape_labels,
                            fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
    classifier.load()
    ourVisualiser = Visualiser()
    ourVisualiser.load()

    approx_nn_model = AnnoyIndex(vector_length, distance_mode)
    approx_nn_model.load(learnt_data_space)

    with open(text_embeddings_file, 'rb') as file:
        embeddings_model = pickle.load(file)

    visual_features = np.array(
        returnVisualFeatures(ourVisualiser, query_image))
    text_query_v_unw = np.array(
        get_query_vector2(query_text, embeddings_model))
    text_query_vector = w * \
        np.array(get_query_vector2(query_text, embeddings_model))
    labels = np.array(get_query_vector2(
        " ".join(returnTextWords(classifier, query_image)), embeddings_model))
    one_hot_encoded_vectors = np.array(
        returnOneHot(classifier, query_image))

    concatenated_array = np.concatenate(
        (visual_features, text_query_vector, labels, one_hot_encoded_vectors), axis=0)

    # To be used for evaluating cosine similarity
    query_vector_unweighted = np.concatenate(
        (visual_features, text_query_v_unw, labels, one_hot_encoded_vectors), axis=0)

    with open(training_dict_files, 'rb') as file:
        images_list = pickle.load(file)

    nearest_indices = approx_nn_model.get_nns_by_vector(
        concatenated_array, n=number_retrieved)
    print(nearest_indices)

    # Retrieve the nearest words based on the indices
    nearest_images = [os.path.join(train_folder, images_list[index])
                      for index in nearest_indices]
    nearest_image_vectors = [approx_nn_model.get_item_vector(
        key) for key in nearest_indices]
    avg_angle_score = ang_avg(query_vector_unweighted, nearest_image_vectors)
    print("Final score efficiency = ", avg_angle_score)

    return nearest_images


# query_img = "datasets/DeepFashion/images/MEN-Denim-id_00000080-01_7_additional.jpg"
# query_text = "sweater denim pants"


# print(Nearest_images(
#     query_image=query_img,
#     query_text=query_text,
#     w=100000
# ))
