import os
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, Tensor, load, no_grad
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

# Imposta il dispositivo (cuda se disponibile, altrimenti cpu)
device = device("cuda" if is_available() else "cpu")

def somma_valori_posizioni(lista):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for sublist in lista:
        if len(sublist) >= 2:  # Assicurati che ci siano almeno tre elementi nella sottolista
            sum1 += sublist[0] 
            sum2 += sublist[1]
            sum3 += sublist[2]  # Somma gli elementi alle posizioni 1, 2 e 3 (0-based)
    return (sum1, sum2, sum3)

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

class MyClassifierModel(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out = self.Backbone(image)
        return out

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
TEST_PATH = "/mnt/sdc1/mmarseglia/dataset2_effective/test"

# Leggi il test set
test_data = pd.read_csv(
    '/user/mmarseglia/second_split_60_30_10/test_dataset2_single_head.csv',
    names=["image", "label", "class"],
    dtype={'image': 'str', 'label': 'str', 'class': 'str'}
)
X_test = test_data["image"][1:]
y_test = test_data["label"][1:]
classes = test_data["class"][1:]

# Crea un dataset per il test
df_test = pd.DataFrame({'image': X_test, 'label': y_test})
testDataSet = dataFrameDataset(df_test, TEST_PATH, data_transforms_test)

# Define a function to perform your operations on a single file
def process_single_file(file_path):
    
    # For a model pretrained on VGGFace2
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    ClassifierModel = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 27),
    )

    # Carica il modello allenato
    model = MyClassifierModel(backbone, ClassifierModel)
    model.load_state_dict(load(file_path))
    model.to(device)
    model.eval()
    
    # Crea un DataLoader per il test
    batch_size = 128  # Imposta il batch size per il test
    test_set = DataLoader(testDataSet, shuffle=False, batch_size=batch_size, num_workers=3)

    correct = 0
    correct_per_class = (0,0,0)
    total = 0
    true_labels = []
    predicted_labels = []
    two_on_three_acc = 0

    with no_grad():
        for sample_batched in tqdm(test_set):
            inputs = sample_batched['image'].float().to(device)
            labels = sample_batched['label'].long().to(device)

            outputs = model(inputs)
            prob = nn.Softmax(dim=1)(outputs)
            predictions = argmax(prob, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Lista delle stringhe corrispondenti agli indici del tensore
            pred_strings = [all_classes[index].split("-") for index in predictions]
            lab_strings = [all_classes[index].split("-") for index in labels]

            classication_per_subclass = []
            
            for i in range(min(len(pred_strings), len(lab_strings))):
                sublista1 = pred_strings[i]
                sublista2 = lab_strings[i]
                nuova_sublista = []

                for j in range(3):
                    if j < len(sublista1) and j < len(sublista2):
                        nuova_sublista.append(1 if sublista1[j] == sublista2[j] else 0)
                    elif j > len(sublista1)-1 and j > len(sublista2)-1:
                        nuova_sublista.append(1)
                    else:
                        nuova_sublista.append(0)

                classication_per_subclass.append(nuova_sublista)
                if __builtins__.sum(nuova_sublista) > 1:
                    two_on_three_acc += 1
            correct_per_class = tuple(x + y for x, y in zip(correct_per_class , somma_valori_posizioni(classication_per_subclass)))
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

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

    # Salva la figura
    plt.savefig('/user/mmarseglia/second_split_60_30_10/all_matrices/'+file_name+'.png')

    # Calcola l'accuratezza
    accuracy = 100 * correct / total
    acc_per_class =  tuple((x/y)*100 for x, y in zip(correct_per_class , (total, total, total)))
    tt_acc = (two_on_three_acc/total)*100



    # Return the results along with the file name
    return {
        'File Name': file_name,
        'General Accuracy': accuracy,
        'Two on Three Accuracy': tt_acc,
        'Accuracy per Class': acc_per_class
    }

# Define the directory containing your .pt files
directory_path = '/user/mmarseglia/second_split_60_30_10/classificatore_single_head'

# Create an empty list to store results for each file
results_list = []

# Iterate through all .pt files in the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".pt"):
            print("sto analizzando il file :" + file)
            # Construct the full file path
            file_path = os.path.join(root, file)

            # Process the file and get results
            results = process_single_file(file_path)

            # Append the results to the list
            results_list.append(results)

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results_list)

# Save the DataFrame to a CSV file
results_df.to_csv('results.csv', index=False)
