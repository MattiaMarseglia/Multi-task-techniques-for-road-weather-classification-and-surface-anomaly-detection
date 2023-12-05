import torch
from torch.utils.data import Dataset, DataLoader
import tfrecord

# Definizione della classe per il dataset
class WaymoDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = tfrecord.Reader(file_path)
        self.parse_fn = self.get_parse_fn()

    def __len__(self):
        # Ritorna il numero di campioni nel dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Estrae un singolo campione dal dataset
        example = self.dataset[idx]
        return self.parse_fn(example)

    def get_parse_fn(self):
        # Definizione della funzione di parsing per i dati
        def parse_data(example):
            image = example['image'].numpy().reshape([height, width, channels])
            label = example['label']
            return image, label
        return parse_data

# Definizione dei parametri per il set di dati
batch_size = 32
height = 1024
width = 1280
channels = 3

# Percorso del file TFRecord contenente i dati
file_path = 'path/to/waymo/dataset/file.tfrecord'

# Creazione del set di dati a partire dal file TFRecord
dataset = WaymoDataset(file_path)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Iterazione sul set di dati
for images, labels in dataloader:
    # Elaborazione dei dati del batch
    print(labels)
