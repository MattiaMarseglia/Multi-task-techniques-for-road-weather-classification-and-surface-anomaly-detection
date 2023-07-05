#This code is to try a single regressor with our backbone
import os
import cv2
import dlib
import json
import torch
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchscan import summary #network summary
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import train_test_split
# from torchsummary import summary #network summary
from torchvision.models import Inception_V3_Weights, Inception3
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        age = self.df.iloc[idx,1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'age': age}
        return sample

def preprocess_input(img):
    return  img

class allignment(object):
  
  def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.
        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

        self.randomBrightness = transforms.ColorJitter(brightness=(0.9,1.1))



  def shape_to_normal(self, shape):
      shape_normal = []
      for i in range(0, 5):
          shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
      return shape_normal
  def get_eyes_nose_dlib(self, shape):
      nose = shape[4][1]
      left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
      left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
      right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
      right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
      return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)
  def cosine_formula(self, length_line1, length_line2, length_line3):
      cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
      return cos_a
  def distance(self, a, b):
      return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
  def rotate_point(self, origin, point, angle):
      ox, oy = origin
      px, py = point

      qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
      qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
      return qx, qy


  def is_between(self, point1, point2, point3, extra_point):
      c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
      c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
      c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
      if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
          return True
      else:
          return False

  def load_image(self, path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

  def allignImage(self, img):       
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = self.detector(gray, 0)
    angle = 0
    try:
        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = self.predictor(gray, rect)
            shape = self.shape_to_normal(shape)
            nose, left_eye, right_eye = self.get_eyes_nose_dlib(shape)
            center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            center_pred = (int((x + w) / 2), int((y + y) / 2))
            length_line1 = self.distance(center_of_forehead, nose)
            length_line2 = self.distance(center_pred, nose)
            length_line3 = self.distance(center_pred, center_of_forehead)
            cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
            try:
                angle = np.arccos(cos_a)
            except:
                angle = 0
            rotated_point = self.rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
            if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)
    except:
        pass
    img = Image.fromarray(img)
    return np.array(img.rotate(angle))
    

modelPath = ('./landmarks5.dat')

allignment = allignment(modelPath)

class face_alignment_train(object):

    def __call__(self,img):
        
        if bool(random.getrandbits(1)): 
            img = allignment.randomBrightness(img)

        img = np.array(img) 
        # Convert RGB to BGR 
        img = allignment.allignImage(img).astype(np.uint8)

        if random.random() > 0.7:
            img = cv2.flip(img, 1)

        
        angle = 0 
        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype(np.uint8))

        if bool(random.getrandbits(1)): 
            if bool(random.getrandbits(1)): 
                angle = 5
            else:
                angle = -5
        return preprocess_input(img.rotate(angle))

class face_alignment_val(object):

    def __call__(self,img): 
        
        img = np.array(img) 

        img = allignment.allignImage(img).astype(np.uint8)
        
        return preprocess_input(img)

df = pd.read_csv(
   '/user/2022_va_gr12/training_caip_contest.csv',
    names=["image", "age"],dtype={'image':'str','age':'float'})
df.head()

X_train , X_val, y_train, y_val = train_test_split(df['image'],df['age'],train_size=0.74,random_state=2022, shuffle=True,stratify=df['age'])

df_train = pd.DataFrame({'image':X_train,'age':y_train})
df_val = pd.DataFrame({'image':X_val,'age':y_val})


import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

data_transforms_train = transforms.Compose([
transforms.Resize((160,160)), 
face_alignment_train(),
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

data_transforms_val = transforms.Compose([
transforms.Resize((160,160)), 
face_alignment_val(),
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

PATH = "/mnt/sdc1/2022_va_gr12/training_caip_contest"

trainDataSet = dataFrameDataset(df_train,PATH,data_transforms_train)
valnDataSet = dataFrameDataset(df_val,PATH,data_transforms_val)
batch_size = 256
# create batches

def flattening(array,flattening_coeff):
    mean = np.mean(array)
    diff_array = np.absolute(np.subtract(array,mean))
    rescaled_diff_array = np.multiply(diff_array,flattening_coeff)
    idx = array > mean
    rescaled_diff_array[idx]*=-1
    out = array + rescaled_diff_array
    factor = np.divide(np.min(array),np.min(out))
    final_arr = np.multiply(out,factor)
    return final_arr

int_ages = [int(age) for age in y_train]
counts,bins = np.histogram(int_ages, 81) #how much samples are there for each age 
kern = cv2.getGaussianKernel(11,1)
filtered = np.convolve(counts, kern[:,0], mode= 'same')
class_weights = (1/filtered)*10000
class_weights = torch.Tensor(flattening(class_weights,0.4))
class_weights = class_weights/torch.min(class_weights)


train_set = DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=15)
val_set = DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=15)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}


class eoClassifierModel(nn.Module):
    def __init__(self, Backbone, LasteoClassifierModel):
        super(eoClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.LasteoClassifierModel = LasteoClassifierModel
        
    def forward(self, image):
        featureVect = self.Backbone(image)
        output = self.LasteoClassifierModel(featureVect)
        return output

#DEFINE MODEL

#BACKBONE:
from facenet_pytorch import InceptionResnetV1

# For a model pretrained on VGGFace2
backbone = InceptionResnetV1(pretrained='vggface2')

num_ftrs = backbone.last_linear.out_features

#CLASSIFIER
ClassifierModel = nn.Sequential(
    nn.Linear(num_ftrs, num_ftrs),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(num_ftrs, int(num_ftrs/2)),
    nn.BatchNorm1d(num_features = int(num_ftrs/2)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/2), int(num_ftrs/4)),
    nn.BatchNorm1d(int(num_ftrs/4)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/4), 81),
)

#FINAL MODEL
model = eoClassifierModel(backbone, ClassifierModel)

count = 0
for param in model.Backbone.parameters(): 
    param.requires_grad = False
    count+=1
print(count)

count = 0
for param in model.Backbone.mixed_6a.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.repeat_2.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.mixed_7a.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.repeat_3.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.block8.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.avgpool_1a.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.last_linear.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.Backbone.last_bn.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.LasteoClassifierModel.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

model = model.to(device)


model.eval()
summary(model, (3,160,160))

#considearndo il fatto dei batch bilanciati possiamo andare a calcoalre la MAE e l'AAR del batch e rendersi conto della situazione 
def ClassificationLoss(y_pred,y_true, preds, base_criterion):
    y_true=y_true-1
    #return weighted_loss
    diff = (1 + 0.35*abs(y_true-preds)).to(device)

    y_true=y_true.type(torch.LongTensor).to(device)
    return (base_criterion(y_pred,y_true)*diff).mean()

# Observe that all parameters are being optimized
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
# optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),  lr=1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1e-4)

base_criterion = nn.CrossEntropyLoss(class_weights, reduction='none').to(device)


import time
import copy
import matplotlib.pyplot as plt


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

    toPrint = f'Ciao , sto per allenare {PARAMETERS_AND_NAME_MODEL}'
    print(toPrint)
    possibleChoise = torch.arange(1,82).unsqueeze(0).to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_sigma = 0
            running_mMAE = 0

            # Iterate over data.
            for sample_batched in tqdm(dataloaders[phase]):

                inputs = sample_batched['image'].float().to(device)
                labels = sample_batched['age'].float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs, one_hot_input)
                    outputs = model(inputs)
                    # loss = criterion(outputs, labels)
                    prob = nn.Softmax(dim=1)(outputs)
                    # Classifier EV: prediction = torch.sum(torch.mul(possibleChoise ,prob),dim=1) -> Classifier argmax: prediction = torch.argmax(prob,dim=1)
                    prediction = torch.argmax(prob,dim=1)
                    loss = ClassificationLoss(outputs, labels, prediction, base_criterion)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                prediction += 1
                tmp = (torch.logical_and(prediction >= labels.data-1, prediction <= labels.data+1)).long()
                running_corrects += torch.sum(tmp)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            epoch_acc = ((running_corrects / (dataset_sizes[phase]/batch_size))/batch_size)*100
            epoch_acc = epoch_acc.item()

            toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Epochs {epoch}, {phase} Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
            print(toPrint)


            if phase == 'val':
                val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})
                if early_stopper.early_stop(epoch_loss) == True:
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\nStopped at epoch {epoch}')
                    print(f'Best val Acc: {best_loss:4f}')
                    print(f'Best val Acc: {acc_best_loss:4f}')
                    model.load_state_dict(best_model_wts)
                    torch.save(model.state_dict(), './BoostingSeed/'+ PARAMETERS_AND_NAME_MODEL+'_updated.pt')
                    return model,best_loss 
            else:    
                train_losses.append({'TrainLoss': epoch_loss, 'Valacc': epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                acc_best_loss = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './BoostingSeed/'+ PARAMETERS_AND_NAME_MODEL+'_updated.pt')

        print()

    time_elapsed = time.time() - since
    
    
    toPrint = f'Training of {PARAMETERS_AND_NAME_MODEL} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(toPrint)
    
    toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Best val loss: {best_loss:4f}, Best val Acc: {acc_best_loss:4f}'
    print(toPrint)
    

    with open('./EvaluationCurves/'+PARAMETERS_AND_NAME_MODEL+'_updated'+ str(numTrain) +'training.json', 'w') as f:
        dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
        json.dump(dict, f)
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_loss,train_losses,val_losses
    

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
early_stopper = EarlyStopper(patience=20, min_delta=0.12)
best_loss = 100000
model_ft,best_loss,train_losses,val_losses = train_model(model, base_criterion, optimizer, exp_lr_scheduler,
                       num_epochs=30,best_loss=best_loss,early_stopper=early_stopper,numTrain=1)