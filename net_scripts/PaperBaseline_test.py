#This code is to try a single regressor with our backbone
import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

PARAMETERS_AND_NAME_MODEL = "Baseline EfficientNetBo"

# from torchsummary import summary #network summary
print(torch.__version__)
print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)     

class dataFrameDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        champ_class = self.df.iloc[idx,1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': int(champ_class)}
        return sample

test_data = pd.read_csv(
   '/home/mattia/Desktop/Tesi/dataset_reordered/test.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_test = test_data["image"][1:128*5]
y_test = test_data["label"][1:128*5]
classes = test_data["class"][1:128*5]

# X_test , X_val, y_test, y_val = test_test_split(df['image'],df['age'],test_size=0.74,random_state=2022, shuffle=True,stratify=df['age'])


df_test = pd.DataFrame({'image':X_test,'label':y_test})

data_transforms_test = transforms.Compose([
transforms.ToTensor(),
# transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

TEST_PATH = "/home/mattia/Desktop/Tesi/dataset_reordered/test"


TestDataSet = dataFrameDataset(df_test,TEST_PATH,data_transforms_test)

# create batches
batch_size = 128

testSet = DataLoader(TestDataSet, shuffle=False,batch_size=batch_size)

dataloaders = {'test':testSet}
dataset_sizes = {'test':len(TestDataSet)}


def evaluate(loader, model):
    # initialize metric
    model.eval()

    my_dict = pd.DataFrame()
    running_corrects = 0
    with torch.no_grad():
        
        for sample_batched in tqdm(loader):
            inputs = sample_batched['image'].float().to(device)
            labels = sample_batched['label'].to(device)


            outputs = model(inputs)
            prob = nn.Softmax(dim=1)(outputs)
            prediction = torch.argmax(prob,dim=1)
            correctness = (prediction == labels).long()
            running_corrects += torch.sum(correctness)
        epoch_acc = ((running_corrects / (dataset_sizes['test']/batch_size))/batch_size)*100
        print(epoch_acc)
    #         for idx,out in enumerate(outputs):
    #             tmp_dict = pd.DataFrame({'label':champ_class[idx],'value':torch.argmax(out).item()},index=[1])
    #             my_dict = pd.concat([my_dict, tmp_dict])

    # my_dict.to_csv("/home/mattia/Desktop/Tesi/result.csv", index=False, header=False)


class MyClassifierModel(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Classifier = Classifier
        
    def forward(self, image):
        x = self.Backbone(image)
        output = self.Classifier(x)
        return output

class SingleConvNet(nn.Module):
    def __init__(self):
        super(SingleConvNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 3 canali di input, 16 filtri, kernel 3x3
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Riduzione dimensionale con max pooling
        self.fc = nn.Linear(16 * 120 * 180, 27)  # Layer fully-connected per la classificazione

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#DEFINE MODEL

# import torchvision.models as models

# model_names = [name for name in dir(models) if not name.startswith("__")]
# print(model_names)

#BACKBONE:
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v2

# For a model pretrained on VGGFace2
# backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
backbone = mobilenet_v2(pretrained=True)

ClassifierModel = nn.Sequential(
    
    nn.Linear(1000, 27),
    nn.Softmax(),
    )


# #FINAL MODEL
# model = MyClassifierModel(backbone, ClassifierModel)

# Creazione dell'istanza del modello
model = SingleConvNet()
checkpoint1 = torch.load('/home/mattia/Desktop/Tesi/baseline_models/Baseline EfficientNetBo_updated.pt')
checkpoint1.keys()
model.load_state_dict(checkpoint1)

count = 0
for param in model.parameters(): 
    param.requires_grad = False
    count+=1
print(count)

model = model.to(device)

model.eval()
            

evaluate(testSet,model)