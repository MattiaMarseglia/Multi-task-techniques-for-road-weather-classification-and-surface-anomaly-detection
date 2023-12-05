import os
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, Tensor, load, no_grad, logical_and
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
from sklearn.metrics import confusion_matrix
# import debugpy

WHICH_CLASSIFIER = 1

# Imposta il dispositivo (cuda se disponibile, altrimenti cpu)
device = device("cuda" if is_available() else "cpu")

def somma_valori_posizioni(lista):
    sum1 = 0
    for sublist in lista:
        if len(sublist) >= 2:  # Assicurati che ci siano almeno tre elementi nella sottolista
            sum1 += sublist  # Somma gli elementi alle posizioni 1, 2 e 3 (0-based)
    return sum1

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

#DEFINE MODEL
class MyClassifierModel(nn.Module):

    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out1 = self.Backbone(image)
        return out1

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

# Definisci il percorso del test set
TEST_PATH = "/mnt/sdc1/mmarseglia/dataset/test"

# Leggi il test set
test_data = pd.read_csv(
   '/user/mmarseglia/simple_multihead/test_reordered.csv',
    names=["image", "label","class"],dtype={'image':'str','label':'str','class':'str'})
X_test = test_data["image"][1:]
y_test = test_data["label"][1:]
classes = test_data["class"][1:]

# Crea un dataset per il test
df_test = pd.DataFrame({'image': X_test, 'label': y_test})
testDataSet = dataFrameDataset(df_test, TEST_PATH, data_transforms_test)

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

# Define a function to perform your operations on a single file
def process_single_file(file_path, head):
    
    # For a model pretrained on VGGFace2
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    if head == 1:
        ClassifierModel = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 4),
            )
    elif head == 2:
        ClassifierModel = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 6),
            )
    elif head == 3:
        ClassifierModel = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 5),
            )
    # #FINAL MODEL
    model = MyClassifierModel(backbone, ClassifierModel)
    model.load_state_dict(load(file_path))
    model.to(device)
    model.eval()
    
    # Crea un DataLoader per il test
    batch_size = 128  # Imposta il batch size per il test
    test_set = DataLoader(testDataSet, shuffle=False, batch_size=batch_size, num_workers=3)

    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
        
    with no_grad():
        for sample_batched in tqdm(test_set):
            inputs = sample_batched['image'].float().to(device)
            if head == 1:
                labels = sample_batched['label'][0].long().to(device)

            elif head == 2:
                labels = sample_batched['label'][1].long().to(device)

            elif head == 3:
                labels = sample_batched['label'][2].long().to(device)

            #outputs = model(inputs, one_hot_input)
            out = model(inputs)
            # loss = criterion(outputs, labels)
            prob = nn.Softmax(dim=1)(out)

            pred = argmax(prob,dim=1)
    
            correctness = (pred == labels).long()
            
            total += inputs.size(0)
            correct += sum(correctness)
            
            true_labels.extend(np.array(labels.cpu()))
            predicted_labels.extend(np.array(pred.cpu()))

    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Calcola la confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)


    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(30, 20))

    # Crea la heatmap con annotazioni
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes, ax=ax)

    # Imposta i titoli e le etichette degli assi
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    # Salva la figura conf matrix
    plt.savefig('/user/mmarseglia/all_matrices/matrices_for_single_task/'+file_name+'.png')

    # Calcola l'accuratezza
    accuracy = 100 * correct / total



    # Return the results along with the file name
    return {
        'File Name': file_name,
        'General Accuracy': accuracy,
    }

# Define the directory containing your .pt files
directory_path = '/user/mmarseglia/train_only_on_single_task_multihead/second_training'

# Create an empty list to store results for each file
results_list = []

# ###DEBUG###
# try:
#     debugpy.listen(("localhost", 5679))
#     debugpy.wait_for_client()
# except:
#     print("non mi fermo")
#     pass

# Iterate through all .pt files in the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".pt"):
            if "first" in file:
                head = 1
            if "second" in file:
                head = 2
            if "third" in file:
                head = 3
            print("sto analizzando il file :" + file)
            # Construct the full file path
            file_path = os.path.join(root, file)

            # Process the file and get results
            results = process_single_file(file_path, head)

            # Append the results to the list
            results_list.append(results)

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results_list)

# Save the DataFrame to a CSV file
results_df.to_csv('results.csv', index=False)
