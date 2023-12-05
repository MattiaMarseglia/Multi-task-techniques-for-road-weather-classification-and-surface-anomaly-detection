#This code is to try a single regressor with our backbone
import os
import json
import requests
from torch.cuda import device_count, is_available
from torch import load
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchscan import summary #network summary
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyClassifierModel(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out = self.Backbone(image)
        return out

# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weigths=EfficientNet_B0_Weights.IMAGENET1K_V1)

ClassifierModel = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 27),
    nn.Softmax(),
    )

# #FINAL MODEL
model = MyClassifierModel(backbone, ClassifierModel)

#load checkpoints for the classifiers
checkpoint1 = load('./BoostingSeed/primoClassificatoreArgMax_updated.pt')
checkpoint1.keys()
model.load_state_dict(checkpoint1)

print("caricamento riuscito")