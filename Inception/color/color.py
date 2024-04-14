import os
import sys
import pickle
import rembg
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

images_folder_path = "datasets/DeepFashion/images"

color_model_path = 'models/color_model.pkl'
model = pickle.load(open(color_model_path, 'rb'))


def rgb_to_color_name(rgb_tuples):
    return model.predict(rgb_tuples)


def get_dominant_color_ct(image):
    from colorthief import ColorThief
    color_thief = ColorThief(image)
    dominant_color = color_thief.get_palette(quality=2, color_count=5)
    dominant_color = [list(color) for color in dominant_color]
    return dominant_color[1]


def get_dominant_color(image):
    from sklearn.cluster import KMeans
    import numpy as np

    import cv2
    img = cv2.imread(image)
    height, width, dim = img.shape

    img = img[(height//4):(3*height//4), (width//4):(3*width//4), :]
    height, width, dim = img.shape

    img_vec = np.reshape(img, [height * width, dim])

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(img_vec)

    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]

    colors = []
    for cluster_center in kmeans.cluster_centers_[sort_ix]:
        colors.append([int(cluster_center[2]), int(
            cluster_center[1]), int(cluster_center[0])])

    return colors


def get_colors_for_images(folder_path):
    colors = []
    i = 0
    try:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                colors.append(set(rgb_to_color_name(
                    get_dominant_color(file_path))))
                # colors.append(get_dominant_color(file_path))
                i += 1
                print('Processed', i, 'images')

        colors_dict = dict(zip(files, colors))

        return colors_dict

    except OSError as e:
        print(f"Error reading the folder: {e}")


color_dict = get_colors_for_images(images_folder_path)

pickle.dump(color_dict, open(
            "models/Inception/color_list.pkl", "wb"))
print(color_dict)
