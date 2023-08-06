#This code is to try a single regressor with our backbone
import os
import copy
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
# from torchscan import summary 
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, logical_and
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import debugpy

PARAMETERS_AND_NAME_MODEL = "train_multiHead_without_dummy_variables_without_grad_norm_on_new_machine_corrected"
TOKEN = "5823057823:AAE7Uo4nz2GduJVZYDoX_rPrEvmqYJmNUf0"
chatIds = [168283555] #DA LASCIARE SOLTANTO IL MIO

# from torchsummary import summary #network summary
print(device_count())
device = device("cuda" if is_available() else "cpu")
print(device)

def sendToTelegram(toPrint):
    for chat_id in chatIds:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={toPrint}"
        requests.get(url).json()

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
        if is_tensor(idx):
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
   '/user/mmarseglia/train_reordered.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_train = train_data["image"][1:]
y_train = train_data["label"][1:]
classes = train_data["class"][1:]

val_data = pd.read_csv(
   '/user/mmarseglia/val_reordered.csv',
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

TRAIN_PATH = "/mnt/sdc1/mmarseglia/dataset/train"
VAL_PATH = "/mnt/sdc1/mmarseglia/dataset/validation"

trainDataSet = dataFrameDataset(df_train,TRAIN_PATH,data_transforms_train)
valnDataSet = dataFrameDataset(df_val,VAL_PATH,data_transforms_val)

batch_size = 128 #batches 128
# create batches

train_set = DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=3)
val_set = DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=3)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}

#DEFINE MODEL
class MyClassifierModel(nn.Module):

    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        
    def forward(self, image):
        x = self.Backbone(image)
        out1 = self.Classifier_1(x)
        out2 = self.Classifier_2(x)
        out3 = self.Classifier_3(x)
        return out1, out2, out3

# import torchvision.models as models

# model_names = [name for name in dir(models) if not name.startswith("__")]
# print(model_names)

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

count = 0
for param in model.parameters(): 
    param.requires_grad = True
    count+=1
# for param in model.Backbone.classifier.parameters(): 
#     param.requires_grad = True
#     count+=1
print(count)

model = model.to(device)

model.eval()
# Observe that all parameters are being optimized
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),  lr=1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

criterion = nn.CrossEntropyLoss() # two loss function for two task

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

def train_model(model, base_criterion, optimizer, scheduler, early_stopper,num_epochs=25,best_loss=0.0, numTrain=1, acc_best_loss=0.0, class_dictionary=None):
    train_losses=[]
    val_losses=[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = best_loss
    acc_best_loss = acc_best_loss

    toPrint = f'--------------------------------------\nCiao , sto per allenare {PARAMETERS_AND_NAME_MODEL}'
    sendToTelegram(toPrint)
    
    # debugpy.listen(("localhost", 5678))
    # debugpy.wait_for_client()

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample_batched in tqdm(dataloaders[phase]):
                inputs = sample_batched['image'].float().to(device)
                labels1 = sample_batched['label'][0].long().to(device)
                labels2 = sample_batched['label'][1].long().to(device)
                labels3 = sample_batched['label'][2].long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs, one_hot_input)
                    out1, out2, out3 = model(inputs)
                    # loss = criterion(outputs, labels)
                    prob1 = nn.Softmax(dim=1)(out1)
                    prob2 = nn.Softmax(dim=1)(out2)
                    prob3 = nn.Softmax(dim=1)(out3)

                    pred1 = argmax(prob1,dim=1)
                    pred2 = argmax(prob2,dim=1)
                    pred3 = argmax(prob3,dim=1)

                    mask1 = logical_and(logical_and(logical_and(logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                    mask3 = logical_and(logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
                    # Calculate loss for each sample using the masks
                    loss1 = criterion(out1[mask1].squeeze(), labels1[mask1].squeeze())
                    loss2 = criterion(out2, labels2)
                    loss3 = criterion(out3[mask3].squeeze(), labels3[mask3].squeeze())

                    final_loss = loss1 + loss2 + loss3

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        final_loss.backward()
                        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                        optimizer.step()

                # statistics
                running_loss += final_loss.item()
                pred1[~mask1] = labels1[~mask1]
                pred3[~mask3] = labels3[~mask3]
                correctness = (logical_and(logical_and(pred1 == labels1, pred2 == labels2), pred3 == labels3)).long()
                running_corrects += sum(correctness)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            epoch_acc = ((running_corrects / (dataset_sizes[phase]/batch_size))/batch_size)*100
            epoch_acc = epoch_acc.item()

            toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Epochs {epoch}, {phase} Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
            sendToTelegram(toPrint)

            if phase == 'val':
                val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})
                if early_stopper.early_stop(epoch_loss) == True:
                    time_elapsed = time.time() - since
                    sendToTelegram(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\nStopped at epoch {epoch}')
                    sendToTelegram(f'Best val Acc: {best_loss:4f}')
                    sendToTelegram(f'Best val Acc: {acc_best_loss:4f}')

                    model.load_state_dict(best_model_wts)
                    save(model.state_dict(), './baseline_models/'+ PARAMETERS_AND_NAME_MODEL+'.pt')
                    return model,best_loss 
            else:    
                train_losses.append({'TrainLoss': epoch_loss, 'Valacc': epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                acc_best_loss = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save(model.state_dict(), './baseline_models/'+ PARAMETERS_AND_NAME_MODEL+'.pt')
                

    time_elapsed = time.time() - since
    
    
    toPrint = f'Training of {PARAMETERS_AND_NAME_MODEL} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    sendToTelegram(toPrint)
    toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Best val loss: {best_loss:4f}, Best val Acc: {acc_best_loss:4f}'
    sendToTelegram(toPrint)

    

    with open('./baseline_evaluation_curves/'+PARAMETERS_AND_NAME_MODEL+ str(numTrain) +'training.json', 'w') as f:
        dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
        json.dump(dict, f)
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_loss,train_losses,val_losses
    

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
early_stopper = EarlyStopper(patience=20, min_delta=0.12)
best_loss = 100000
model_ft,best_loss,train_losses,val_losses = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=6,best_loss=best_loss,early_stopper=early_stopper,numTrain=1, class_dictionary=class_dictionary)