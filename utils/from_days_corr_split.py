import os
import random
import shutil
import os

def split_dataset(input_dir, output_dir, train_percent, validation_percent, test_percent):
    # Verifica che le percentuali siano valide
    total_percent = train_percent + validation_percent + test_percent
    if total_percent != 100:
        raise ValueError("Le percentuali devono sommare 100.")

    # Crea le cartelle di output se non esistono già
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Leggi la lista dei file nel dataset di input
    file_list = os.listdir(input_dir)
    num_files = 1028301

    # Calcola il numero di file per ogni set
    train_count = int(num_files * (train_percent / 100))
    validation_count = int(num_files * (validation_percent / 100))
    test_count = int(num_files * (test_percent / 100))

    # Imposta il seed per ottenere lo stesso ordine casuale ogni volta
    random.seed(1999)

    # Ottieni una lista casuale dei file
    random.shuffle(file_list)
    train_tmp = 0
    val_tmp = 0
    test_tmp = 0

    # Copia i file nei rispettivi set
    for folder_name in file_list:
        folder_path = os.path.join(input_dir, folder_name)
        print("folder path: ", folder_path)
        for root, dirs, files in os.walk(folder_path):
            if test_tmp + len(files) < test_count:
                output_subdir = 'test'
                test_tmp += len(files)
                print("test completage: ", test_tmp*100/test_count, test_count)
            elif val_tmp + len(files) < validation_count:
                output_subdir = 'validation'
                val_tmp += len(files)
                print("val completage: ", val_tmp*100/validation_count, validation_count)
            else:
                output_subdir = 'train'
                train_tmp += len(files)
                print("train completage: ", train_tmp*100/train_count, train_count)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                output_path = os.path.join(output_dir, output_subdir, file_name)
                shutil.copyfile(file_path, output_path)

        print(f"Copiato {file_name} in {output_subdir}")

# Esempio di utilizzo
input_directory = '/home/mattia/Desktop/Tesi/dataset_into_days'
output_directory = '/home/mattia/Desktop/Tesi/dataset_reordered'
train_percent = 93
validation_percent = 2
test_percent = 5

split_dataset(input_directory, output_directory, train_percent, validation_percent, test_percent)
