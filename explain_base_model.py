#This code is to try a single regressor with our backbone
import torch.nn as nn
from torchscan import summary 
from torch.cuda import device_count, is_available
from torch import device
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# from torchsummary import summary #network summary
print(device_count())
device = device("cuda" if is_available() else "cpu")
print(device)

#DEFINE MODEL
class MyClassifierModel(nn.Module):

    def __init__(self, Backbone, Classifier_1):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        
    def forward(self, image):
        x = self.Backbone(image)
        out1 = self.Classifier_1(x)
        return out1

# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weigths=EfficientNet_B0_Weights.IMAGENET1K_V1)

ClassifierModel_1 = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 4),
    )
# #FINAL MODEL
model = MyClassifierModel(backbone, ClassifierModel_1)

model = model.to(device)

model.eval()

summary(model, (3,360,240))