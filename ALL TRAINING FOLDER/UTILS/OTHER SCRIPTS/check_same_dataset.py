import os
from PIL import Image
import csv
from collections import defaultdict

CREATE_CSV = False
COMPARE_CSV = True

DIRECTORY_DATASET_SPLITTED = "/mnt/sdc1/mmarseglia/dataset"
DIRECTORY_FOR_CSV_FIRST_DATASET = "/user/mmarseglia/all_csv/CSV_10_GB"
DIRECTORY_FOR_CSV_SECOND_DATASET = "/user/mmarseglia/all_csv/CSV_SCHEDA_MIGLIORE"
DIRECTORY_WHERE_SAVE_CSV = "/user/mmarseglia/all_csv/CSV_10_GB"

def save_image_names_to_csv(root_path):
    for root, dirs, files in os.walk(root_path):
        folder_name = os.path.basename(root)
        image_names = []

        for file in files:
            if file.lower().endswith('.jpg'):
                image_names.append(file)

        if image_names:
            csv_file_path = os.path.join(DIRECTORY_WHERE_SAVE_CSV, f"{folder_name}.csv")
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Image Names'])
                csv_writer.writerows([[name] for name in image_names])

def compare_csv_files(directory1, directory2):
    csv_files1 = [file for file in os.listdir(directory1) if file.endswith('.csv')]
    csv_files2 = [file for file in os.listdir(directory2) if file.endswith('.csv')]

    common_csv_files = set(csv_files1) & set(csv_files2)

    if not common_csv_files:
        print("Nessun file CSV comune trovato nelle due directory.")
        return

    image_mapping = defaultdict(list)

    for csv_file in common_csv_files:
        csv_path1 = os.path.join(directory1, csv_file)
        csv_path2 = os.path.join(directory2, csv_file)

        with open(csv_path1, mode='r') as file1, open(csv_path2, mode='r') as file2:
            reader1 = csv.reader(file1)
            reader2 = csv.reader(file2)

            image_names1 = [row[0] for row in reader1 if row]
            image_names2 = [row[0] for row in reader2 if row]

            image_names1.sort()
            image_names2.sort()
            
            image_mapping[csv_file] = image_names1 == image_names2

    return image_mapping

if CREATE_CSV:
    save_image_names_to_csv(DIRECTORY_DATASET_SPLITTED)
if COMPARE_CSV:
    result = compare_csv_files(DIRECTORY_FOR_CSV_FIRST_DATASET, DIRECTORY_FOR_CSV_SECOND_DATASET)

    for csv_file, are_equal in result.items():
        print(f"{csv_file}: {'Stessi' if are_equal else 'Diversi'}")
