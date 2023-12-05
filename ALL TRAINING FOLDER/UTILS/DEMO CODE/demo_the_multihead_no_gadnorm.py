import gradio as gr
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, Tensor, load, no_grad, logical_or, ones
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from torchvision import transforms
import torch
import numpy as np
# import debugpy

#INSERT_IF_YOU_WANT_TO_OBTAIN_PERFORMANCES_ON_TEST_OR_ON_VAL VALUES:
#test
#validation

FILE_PATH = "/home/mattia/Desktop/Tesi/demo/models_weight/multihead_no_gradnorm.pt"
# Imposta il dispositivo (cuda se disponibile, altrimenti cpu)
device = device("cuda" if is_available() else "cpu")

#DEFINE MODEL
class MyClassifierModel(nn.Module):

    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        # self.weights = nn.Parameter(ones(3).float())
        
    def forward(self, image):
        x = self.Backbone(image)
        out1 = self.Classifier_1(x)
        out2 = self.Classifier_2(x)
        out3 = self.Classifier_3(x)
        return out1, out2, out3

all_classes = ["ice",  "fresh_snow",  "melted_snow",  "wet-mud",  "wet-gravel",  "wet-asphalt-slight",
           "wet-asphalt-smooth",  "wet-asphalt-severe",  "wet-concrete-slight",  "wet-concrete-smooth",
           "wet-concrete-severe",  "water-mud",  "water-gravel",  "water-asphalt-slight",  "water-asphalt-smooth",
           "water-asphalt-severe",  "water-concrete-slight",  "water-concrete-smooth",  "water-concrete-severe",
           "dry-mud",  "dry-gravel",  "dry-asphalt-slight",  "dry-asphalt-smooth",  "dry-asphalt-severe",
           "dry-concrete-slight",  "dry-concrete-smooth",  "dry-concrete-severe"]

# Definisci la trasformazione per il test
data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

#DICTIONARY FROM MONOCLASS TO MULTICLASS 
class_dictionary = {
    "0":3-5-4,
    "1":3-3-4,
    "2":3-4-4,
    "3":3-1-2,
    "4":3-1-3,
    "5":1-1-0,
    "6":0-1-0,
    "7":2-1-0,
    "8":1-1-1,
    "9":0-1-1,
    "10":2-1-1,
    "11":3-2-2,
    "12":3-2-3,
    "13":1-2-0,
    "14":0-2-0,
    "15":2-2-0,
    "16":1-2-1,
    "17":0-2-1,
    "18":2-2-1,
    "19":3-0-2,
    "20":3-0-3,
    "21":1-0-0,
    "22":0-0-0,
    "23":2-0-0,
    "24":1-0-1,
    "25":0-0-1,
    "26":2-0-1,
}

class_dictionary_inverted_extended = {
    "3-5-4": "0",
    "3-3-4": "1",
    "3-4-4": "2",
    "3-1-2": "3",
    "3-1-3": "4",
    "1-1-0": "5",
    "0-1-0": "6",
    "2-1-0": "7",
    "1-1-1": "8",
    "0-1-1": "9",
    "2-1-1": "10",
    "3-2-2": "11",
    "3-2-3": "12",
    "1-2-0": "13",
    "0-2-0": "14",
    "2-2-0": "15",
    "1-2-1": "16",
    "0-2-1": "17",
    "2-2-1": "18",
    "3-0-2": "19",
    "3-0-3": "20",
    "1-0-0": "21",
    "0-0-0": "22",
    "2-0-0": "23",
    "1-0-1": "24",
    "0-0-1": "25",
    "2-0-1": "26",
    "not_possible": "27",
}

# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

ClassifierModel_1 = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 4),
    )
ClassifierModel_2 = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 6),
    )
ClassifierModel_3 = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 5),
    )
# #FINAL MODEL
model = MyClassifierModel(backbone, ClassifierModel_1, ClassifierModel_2, ClassifierModel_3)
model.load_state_dict(load(FILE_PATH))
model.to(device)
model.eval()

# Define a function to perform your operations on a single file
def process_single_image(input_image):
    with no_grad():
        
        input = data_transforms_test(input_image).float().to(device)
        input = torch.unsqueeze(input, 0)
        #outputs = model(inputs, one_hot_input)
        out1, out2, out3 = model(input)
        # loss = criterion(outputs, labels)
        prob1 = nn.Softmax(dim=1)(out1)
        prob2 = nn.Softmax(dim=1)(out2)
        prob3 = nn.Softmax(dim=1)(out3)

        pred1 = argmax(prob1,dim=1)
        pred2 = argmax(prob2,dim=1)
        pred3 = argmax(prob3,dim=1)
        if logical_or(logical_or((pred2 == 3 ),  ( pred2 == 4 )),  ( pred2 == 5)):
            pred1 = 3
            pred3 = 4
        elif logical_or(( pred3 == 2 ), ( pred3 == 3)):
            pred1 = 3
        predict = all_classes[int(class_dictionary_inverted_extended[f"{int(pred1)}-{int(pred2)}-{int(pred3)}"])]
        return predict

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(process_single_image, gr.Image(shape=(240, 360)), outputs="text")
demo.launch()