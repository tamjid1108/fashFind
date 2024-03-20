import torch
import torch.optim as optim
import torchvision
from torch import nn
import os
from PIL import Image


model_path = "models/Inception/colored_visualiser.pth"


class Visualiser(nn.Module):
    def __init__(self):
        super(Visualiser, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0',
                                        'inception_v3',
                                        weights="DEFAULT")

        self.inception.fc = nn.Identity()

        self.inception.eval()

        for param in self.inception.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.to(self.device)

    def forward(self, image):
        out = self.inception(image).view(-1)
        return out

    def save(self):
        torch.save(self.state_dict(), model_path)

        print("[SAVE TRAINED MODEL TO]", model_path)
        return

    def load(self):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            print("Model does not exist, training it first")
            self.save()
            self.load()

        print("[LOAD PRETRAINED MODEL FROM]", model_path)
        return

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


sample = "MEN-Denim-id_00000080-01_7_additional.jpg"
img1 = "datasets/DeepFashion/images/MEN-Denim-id_00000080-01_7_additional.jpg"


# returns a list of 2048-length visual features from image_path
def returnVisualFeatures(visualiser: Visualiser, image_path):
    tensor1 = visualiser.preprocess_image(image_path)
    image_results = visualiser.forward(tensor1)
    feature_list = torch.Tensor.tolist(image_results)
    return feature_list


def b():
    visualiser = Visualiser()

    visualiser.load()

    visual_features = returnVisualFeatures(visualiser, img1)

    print(visual_features)
    print(len(visual_features))


if __name__ == "__main__":
    b()
