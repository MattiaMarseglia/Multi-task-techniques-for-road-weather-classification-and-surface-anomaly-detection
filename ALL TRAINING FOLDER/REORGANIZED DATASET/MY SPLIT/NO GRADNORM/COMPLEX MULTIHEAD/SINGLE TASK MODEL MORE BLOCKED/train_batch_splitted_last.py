#This code is to try a single regressor with our backbone
import os
import json
import requests
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, Tensor
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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
# import debugpy

TO_SPLIT = False
HOW_SPLIT = 4
PARAMETERS_AND_NAME_MODEL = "splitted_batch_183_128_batch_new_dataset_split_corrected"
TOKEN = "5823057823:AAE7Uo4nz2GduJVZYDoX_rPrEvmqYJmNUf0"

chatIds = [168283555] #DA LASCIARE SOLTANTO IL MIO
#1407029395,163426269
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
        champ_class = self.df.iloc[idx,1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': int(champ_class)}
        return sample

train_data = pd.read_csv(
   '/user/mmarseglia/second_split_60_30_10/train_dataset2_single_head.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_train = train_data["image"][1:]
y_train = train_data["label"][1:]
classes = train_data["class"][1:]

val_data = pd.read_csv(
   '/user/mmarseglia/second_split_60_30_10/validation_dataset2_single_head.csv',
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
if TO_SPLIT:
    batch_size = int(batch_size/HOW_SPLIT)
    
# create batches

train_set = DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=3)
val_set = DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=3)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}


class MyClassifierModel(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out = self.Backbone(image)
        return out

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


# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


ClassifierModel = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 27),
    )

# #FINAL MODEL
model = MyClassifierModel(backbone, ClassifierModel)

count = 0
for param in model.parameters(): 
    param.requires_grad = True
    count+=1
for param in model.Backbone.features[0].parameters():
    param.requires_grad = False
for param in model.Backbone.features[1].parameters():
    param.requires_grad = False
for param in model.Backbone.features[2].parameters():
    param.requires_grad = False
for param in model.Backbone.features[3].parameters():
    param.requires_grad = False
for param in model.Backbone.features[4].parameters():
    param.requires_grad = False
    
print(model)
trainable_parameters = sum(Tensor([p.numel() for p in model.parameters() if p.requires_grad]))
total_parameters = sum(Tensor([p.numel() for p in model.parameters()]))
print("PARAMETERS: trainable_parameters= ", str(trainable_parameters)," untrainable_parameters= ", str(total_parameters - trainable_parameters), " total_parameters= ", str(total_parameters))

model = model.to(device)

model.eval()
# summary(model, (3,360,240))
# Observe that all parameters are being optimized
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),  lr=1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

reduction = "mean"
if TO_SPLIT:
    reduction = "sum"
base_criterion = nn.CrossEntropyLoss(reduction=reduction).to(device)

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
        
def train_model(model, base_criterion, optimizer, scheduler, early_stopper,num_epochs=25,best_loss=0.0, numTrain=1, acc_best_loss=0.0):
    train_losses=[]
    val_losses=[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = best_loss
    acc_best_loss = acc_best_loss

    toPrint = f'--------------------------------------\nCiao , sto per allenare {PARAMETERS_AND_NAME_MODEL}'
    sendToTelegram(toPrint)
    # ###DEBUG###
    # try:
    #     debugpy.listen(("localhost", 5679))
    #     debugpy.wait_for_client()
    # except:
    #     print("non mi fermo")
    #     pass

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # zero the parameter gradients
            optimizer.zero_grad()
            # Iterate over data.
            for i, sample_batched in enumerate(tqdm(dataloaders[phase])):
                inputs = sample_batched['image'].float().to(device)
                labels = sample_batched['label'].long().to(device)
                # forward
                # track history if only in train
                with set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs, one_hot_input)
                    outputs = model(inputs)
                    # loss = criterion(outputs, labels)
                    prob = nn.Softmax(dim=1)(outputs)

                    prediction = argmax(prob,dim=1)
                    if TO_SPLIT:
                        loss = base_criterion(outputs, labels)/128
                    else:
                        loss = base_criterion(outputs, labels)

                    
                    # backward + optimize only if in training phase
                    
                    if phase == 'train':
                        loss.backward()
                        if TO_SPLIT:
                            if (i+1)%HOW_SPLIT == 0:
                                optimizer.step()
                                # every 10 iterations of batches of size 10
                                optimizer.zero_grad()
                        else:
                            optimizer.step()
                            optimizer.zero_grad()


                # statistics
                running_loss += loss.item()
                correctness = (prediction == labels).long()
                running_corrects += sum(correctness)
            if phase == 'train':
                scheduler.step()
                
            if TO_SPLIT:
                epoch_loss = running_loss / (dataset_sizes[phase]/(batch_size*HOW_SPLIT))
            else:
                epoch_loss = running_loss / (dataset_sizes[phase]/(batch_size))
                
            epoch_acc = (running_corrects / dataset_sizes[phase])*100
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
model_ft,best_loss,train_losses,val_losses = train_model(model, base_criterion, optimizer, exp_lr_scheduler,
                       num_epochs=6,best_loss=best_loss,early_stopper=early_stopper,numTrain=1)