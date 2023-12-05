#This code is to try a single regressor with our backbone
import os
import copy
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from torchscan import summary 
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import debugpy
import torch

import sys
sys.path.append('/user/mmarseglia/second_split_60_30_10/src')

from trainer import Trainer

PARAMETERS_AND_NAME_MODEL = "train_multiHead_without_dummy_variables_with_REAL_grad_norm_new_split_norm_in_epoch_first_time"
# TOKEN = "5823057823:AAE7Uo4nz2GduJVZYDoX_rPrEvmqYJmNUf0"
# chatIds = [168283555] #DA LASCIARE SOLTANTO IL MIO

# from torchsummary import summary #network summary
print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# def sendToTelegram(toPrint):
#     for chat_id in chatIds:
#         url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={toPrint}"
#         requests.get(url).json()

class dataFrameDataset(torch.utils.data.Dataset):
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
        champ_classes = self.df.iloc[idx,1]
        image = Image.open(img_name)
        champ_list = champ_classes.split("-")

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': (int(champ_list[0]),int(champ_list[1]),int(champ_list[2]))}
        return sample

train_data = pd.read_csv(
   '/user/mmarseglia/second_split_60_30_10/train_dataset2_Multihead.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_train = train_data["image"][1:]
y_train = train_data["label"][1:]
classes = train_data["class"][1:]

val_data = pd.read_csv(
   '/user/mmarseglia/second_split_60_30_10/validation_dataset2_Multihead.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_val = val_data["image"][1:]
y_val = val_data["label"][1:]
classes = val_data["class"][1:]

print(len(X_train), len(y_train))
print(len(X_val), len(y_val))
# X_train , X_val, y_train, y_val = train_test_split(df['image'],df['age'],train_size=0.74,random_state=2022, shuffle=True,stratify=df['age'])


df_train = pd.DataFrame({'image':X_train,'label':y_train})
df_val = pd.DataFrame({'image':X_val,'label':y_val})

data_transforms_train = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

data_transforms_val = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

TRAIN_PATH = "/mnt/sdc1/mmarseglia/dataset2_effective/train"
VAL_PATH = "/mnt/sdc1/mmarseglia/dataset2_effective/validation"

trainDataSet = dataFrameDataset(df_train,TRAIN_PATH,data_transforms_train)
valnDataSet = dataFrameDataset(df_val,VAL_PATH,data_transforms_val)

batch_size = 128 #batches 128
# create batches

train_set = torch.utils.data.DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=3)
val_set = torch.utils.data.DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=3)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}

#DEFINE MODEL
class MyClassifierModel(torch.nn.Module):

    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = torch.nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        self.weights = torch.torch.torch.nn.Parameter(torch.ones(3).float())

        
    def forward(self, image):
        x = self.Backbone(image)
        out1 = self.Classifier_1(x)
        out2 = self.Classifier_2(x)
        out3 = self.Classifier_3(x)
        return out1, out2, out3
    
    def get_last_shared_layer(self):
        return self.Backbone.features[8][0]

# import torchvision.models as models

# model_names = [name for name in dir(models) if not name.startswith("__")]
# print(model_names)

# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

ClassifierModel_1 = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1280, 4),
    )
ClassifierModel_2 = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1280, 6),
    )
ClassifierModel_3 = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1280, 5),
    )
# #FINAL MODEL
model = MyClassifierModel(backbone, ClassifierModel_1, ClassifierModel_2, ClassifierModel_3)

count = 0
for param in model.parameters(): 
    param.requires_grad = True
    count+=1
# for param in model.Backbone.classifier.parameters(): 
#     param.requires_grad = True
#     count+=1
print(model)

trainable_parameters = sum(torch.Tensor([p.numel() for p in model.parameters() if p.requires_grad]))
total_parameters = sum(torch.Tensor([p.numel() for p in model.parameters()]))
print("PARAMETERS: trainable_parameters= ", str(trainable_parameters)," untrainable_parameters= ", str(total_parameters - trainable_parameters), " total_parameters= ", str(total_parameters))

model = model.to(device)

model.eval()
# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),  lr=1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

criterion = torch.nn.CrossEntropyLoss() # two loss function for two task

import time
import copy

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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

trainer = Trainer(model, PARAMETERS_AND_NAME_MODEL, dataloaders, dataset_sizes, batch_size)
to_print = "sto per addestrare: " + PARAMETERS_AND_NAME_MODEL
# sendToTelegram(to_print)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
early_stopper = EarlyStopper(patience=5, min_delta=0.12)
best_loss = 100000
model_ft,best_loss,train_losses,val_losses = trainer.train_norm_in_epoch(criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=8,best_loss=best_loss,early_stopper=early_stopper,numTrain=1)