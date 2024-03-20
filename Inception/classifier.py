import torch
import torch.optim as optim
import torchvision
from torch import nn
import os
from PIL import Image
from math import ceil

'''
Classifier Object has the following methods
learn(image_dir) -> to learn on a given database
save() -> to save state to "classifier.pth"
load() -> to load state from "classifier.pth"
forward(image_tensor) -> to yield output 18-length attributes tensor
preprocess_image(image_path) -> to read file and construct tensor
'''

img_location = "datasets/DeepFashion/images"
shape_labels = "datasets/DeepFashion/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "datasets/DeepFashion/labels/texture/fabric_ann.txt"
pattern_texture_labels = "datasets/DeepFashion/labels/texture/pattern_ann.txt"

model_path = "models/Inception/classifier.pth"

# Returns mapping from ImageName -> Features List


def read_data(filename):
    labels_out = {}
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    for line in lines:
        words = line.split()
        img_name = words[0]
        shape_vector = []
        for word in words[1:]:
            if word != "NA":
                shape_vector.append(int(word))
            else:
                shape_vector.append(-1)
        labels_out[img_name] = shape_vector
    return labels_out

# shape_labels_dict = read_data(shape_labels)
# fabric_texture_labels_dict = read_data(fabric_texture_labels)
# pattern_texture_labels_dict = read_data(pattern_texture_labels)


class Classifier(nn.Module):
    def __init__(self, shape_labels_file,  fabric_texture_file, pattern_file):
        super(Classifier, self).__init__()

        # Set the number of output classes
        self.num_classes = 18

        # Load the Inception V3 model with default weights
        self.inception = torch.hub.load(
            'pytorch/vision:v0.10.0',  # Load from the PyTorch model zoo
            'inception_v3',  # Load the Inception V3 architecture
            weights="DEFAULT"  # Load with default pre-trained weights
        )

        # Define the last linear layer for the custom classifier
        self.last_layer = torch.nn.Linear(
            # Set the input size to the number of features in the original Inception V3 classifier
            self.inception.fc.in_features,
            self.num_classes  # Set the output size to the number of classes in the classification task
        )

        # Replace the fully connected (fc) layer of the Inception V3 model with an Identity layer
        self.inception.fc = nn.Identity()

        # Set the Inception V3 model to evaluation mode
        self.inception.eval()

        for param in self.inception.parameters():
            param.requires_grad = False
        for param in self.last_layer.parameters():
            param.requires_grad = True

        # Load the shape, fabric texture, and pattern texture labels
        self.shape_labels_dict = read_data(shape_labels_file)
        self.fabric_texture_labels_dict = read_data(fabric_texture_file)
        self.pattern_texture_labels_dict = read_data(pattern_file)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.to(self.device)

    def forward(self, image):
        # print("Image shape = ", image.shape)
        x = self.inception(image)
        out = self.last_layer(x).view(-1)
        return out

    # Takes in full path!
    def preprocess_image(self, image_filename):
        input_image = Image.open(image_filename)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def save(self):
        torch.save(self.state_dict(), model_path)

        print("[SAVE TRAINED MODEL TO]", model_path)
        return

    def load(self):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            print("Model does not exist, training it first")
            self.learn(img_location)
            self.save()
            self.load()

        print("[LOAD PRETRAINED MODEL FROM]", model_path)
        return

    def learn(self, image_dir):
        file_list = os.listdir(image_dir)
        file_list_with_path = [os.path.join(
            image_dir, file) for file in file_list]
        j = 0
        for i in range(len(file_list)):
            # for i in range(500):
            img_file_name = file_list[i]
            img_file_path = file_list_with_path[i]

            if img_file_name in self.shape_labels_dict.keys() and img_file_name in self.fabric_texture_labels_dict.keys() and img_file_name in self.pattern_texture_labels_dict.keys():
                vector_img = self.shape_labels_dict[img_file_name] + \
                    self.fabric_texture_labels_dict[img_file_name] + \
                    self.pattern_texture_labels_dict[img_file_name]
                vector_tensor = torch.tensor(vector_img, dtype=torch.float).to(
                    self.device)  # The 23 length vector

                image_processed = self.preprocess_image(
                    img_file_path).to(self.device)

                model_output = self.forward(image_processed)

                # print("Model's output has dims = ", model_output.shape)
                # print("Tensor to train to has dims = ", vector_tensor.shape)
                self.optimizer.zero_grad()
                loss = self.criterion(model_output, vector_tensor)
                loss.backward()
                self.optimizer.step()
            else:
                j += 1
            if i % 100 == 0:
                print("Trained on images = ", i)

        print("Total unbalanced keys = ", j)
        return


def returnOneHot(ourClassifier: Classifier, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor))
    inferres_ints = [round(x) for x in inferred_features]

    features_length_dataset = [6, 5, 4, 3, 5,
                               3, 3, 3, 5, 7, 3, 3, 8, 8, 8, 8, 8, 8]
    for i in range(len(inferres_ints)):  # Prune
        if inferres_ints[i] < 0:
            inferres_ints[i] = 0
        elif inferres_ints[i] >= features_length_dataset[i]:
            inferres_ints[i] = features_length_dataset[i] - 1

    one_hot = [0 for _ in range(sum(features_length_dataset))]
    #  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,

    cur_ind = 0
    for i in range(18):
        # We're talking about feature ith
        # print("i = ", i, "should have feature starting at = ", cur_ind, " and now, we have entry at =", inferres_ints[i])

        # If in valid range, turn this indicator on
        one_hot[cur_ind + inferres_ints[i]] = 1
        # print("writing 1 to position : ", cur_ind + inferres_ints[i])

        # Increment pointer by len(feature)
        cur_ind += features_length_dataset[i]

    return one_hot


def returnTextWords(ourClassifier: Classifier, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor))
    inferres_ints = [round(x) for x in inferred_features]
    features_length_dataset = [6, 5, 4, 3, 5,
                               3, 3, 3, 5, 7,
                               3, 3, 8, 8, 8,
                               8, 8, 8]
    for i in range(len(inferres_ints)):  # Prune
        if inferres_ints[i] < 0:
            inferres_ints[i] = 0
        elif inferres_ints[i] >= features_length_dataset[i]:
            inferres_ints[i] = features_length_dataset[i] - 1

    # print("Inferred Feautures = ", inferres_ints)
    words = []
    #  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
    dictionary_features = {
        0: {0: "sleeveless", 1: "short-sleeve", 2: "medium-sleeve", 3: "long-sleeve"},
        1: {0: "three-point lower cloth length", 1: "medium short lower cloth length", 2: "three-quarter lower cloth length", 3: "long leggings lower cloth length"},
        2: {1: "socks", 2: "leggings"},  # Socks
        3: {1: "hat"},  # Hat
        4: {1: "eyeglasses", 2: "sunglasses"},  # Glasses
        5: {1: "neckwear"},  # Neackwear
        6: {1: "wrist wearing"},  # Wrist wearing
        7: {1: "ring"},  # ring
        # Waist accesories
        8: {1: "belt", 2: "clothing on waist", 3: "hidden waist"},
        9: {0: "V shape neckline", 1: "square neckline", 2: "round neckline", 3: "standing neckline", 4: "lapel neckline", 5: "suspenders neckline"},  # Neckline
        10: {0: "cardigan"},  # Cardigan?
        11: {0: "navel not covered", 1: "navel covered"},  # Navel Covered?
        # Upper Fabric Annotations
        12: {0: "denim upper fabric", 1: "cotton upper fabric", 2: "leather upper fabric", 3: "furry upper fabric", 4: "knitted upper fabric", 5: "chiffon upper fabric"},
        # Lower Fabric Annotations
        13: {0: "denim lower fabric", 1: "cotton lower fabric", 2: "leather lower fabric", 3: "furry lower fabric", 4: "knitted lower fabric", 5: "chiffon lower fabric"},
        # Outer Fabric Annotations
        14: {0: "denim outer fabric", 1: "cotton outer fabric", 2: "leather outer fabric", 3: "furry outer fabric", 4: "knitted outer fabric", 5: "chiffon upper fabric"},

        15: {0: "floral upper color", 1: "graphic upper color", 2: "striped upper color", 3: "pure upper color", 4: "lattice upper color", 6: "block upper color"},  # Upper Color
        16: {0: "floral lower color", 1: "graphic lower color", 2: "striped lower color", 3: "pure lower color", 4: "lattice lower color", 6: "block lower color"},  # Lower Color
        17: {0: "floral outer color", 1: "graphic outer color", 2: "striped outer color", 3: "pure outer color", 4: "lattice outer color", 6: "block outer color"},  # Outer Color
    }

    for i in range(18):
        # We're talking about feature ith
        if i not in [2, 6, 7]:  # Not counting socks, wrist and ring
            # If encounters valid textual feature
            if inferres_ints[i] in dictionary_features[i].keys():
                words += dictionary_features[i][inferres_ints[i]].split()

    return words


def a():
    ourClassifier = Classifier(shape_labels_file=shape_labels,
                               fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
    ourClassifier.load()

    sample = "MEN-Denim-id_00000080-01_7_additional.jpg"
    img1 = "datasets/DeepFashion/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    img1_t = ourClassifier.preprocess_image(img1)
    print("FORWARD +++++++", ourClassifier.forward(img1_t))
    print("labels shape, fabric, pattern",
          ourClassifier.shape_labels_dict[sample], ourClassifier.fabric_texture_labels_dict[sample], ourClassifier.pattern_texture_labels_dict[sample])
    one_hot = returnOneHot(ourClassifier, img1)
    text_words = returnTextWords(ourClassifier, img1)
    # print("one_hot encoding = ", one_hot)
    print("text words =", text_words)
    return


if __name__ == "__main__":
    a()
