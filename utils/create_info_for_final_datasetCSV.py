# !/usr/bin/python
########## FILE NAMES ##########
### TRAIN LABEL ###
#   202201261342122-dry-asphalt-severe
### TEST ###
#   202201252342299-dry-asphalt-smooth 
### VALIDATION ###
#   202205171728-dry-concrete-smooth
import os
import csv

All_files = []

def scrivi_su_csv(lista, nome_file):
    with open(nome_file, 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for elemento in lista:
            writer.writerow(elemento)

# Esempio di utilizzo


def classes_in_the_subset(cartella_base):
    i = 0
    for dirpath, dirnames, filenames in os.walk(cartella_base):
        print(i,", ",dirnames, len(filenames))
        if i == 0:
            folders = dirnames
            all_set = {"train": [], "validation": [], "test": []}
        else:
            
            print(len(filenames))
            for filename in filenames:
                sub_strings = filename.split("-")
                champ_class = str(sub_strings[1:])
                if champ_class not in all_set[folders[i-1]]:
                    all_set[folders[i-1]].append(champ_class)
        i += 1
    print(all_set)
    print(len(all_set["train"]))
    print(len(all_set["validation"]))
    print(len(all_set["test"]))
    return All_files

def create_subset_CSV(cartella_base, class_dictionary):
    i = 0
    for dirpath, dirnames, filenames in os.walk(cartella_base):
        print(i,", ",dirnames, len(filenames))
        if i == 0:
            folders = dirnames
            info_set = {"train": [("image", "class")], "validation": [("image", "class")], "test": [("image", "class")]}
        else:
            for filename in filenames:
                sub_strings = filename.split("-")
                champ_class = sub_strings[1:]
                class_string = ""
                for elem in champ_class:
                    class_string += elem + "-"
                class_string = class_string.split(".")[0]
                if class_string in class_dictionary.keys():
                    label = class_dictionary[class_string]
                else:
                    print("questa non sta nel dizionario: ",class_string)
                    input()

                info_set[folders[i-1]].append((filename, label, class_string))
        i += 1
    print(len(info_set["train"]))
    print(len(info_set["validation"]))
    print(len(info_set["test"]))
    print(len(info_set["train"]) + len(info_set["validation"]) + len(info_set["test"]))
    return info_set

# Esempio di utilizzo:
cartella_base = "/home/mattia/Desktop/Tesi/dataset_reordered"
class_dictionary = {
    "ice":19,
    "fresh_snow":1,
    "melted_snow":2,
    "wet-mud":3,
    "wet-gravel":4,
    "wet-asphalt-slight":5,
    "wet-asphalt-smooth":6,
    "wet-asphalt-severe":7,
    "wet-concrete-slight":8,
    "wet-concrete-smooth":9,
    "wet-concrete-severe":10,
    "water-mud":11,
    "water-gravel":12,
    "water-asphalt-slight":13,
    "water-asphalt-smooth":14,
    "water-asphalt-severe":15,
    "water-concrete-slight":16,
    "water-concrete-smooth":17,
    "water-concrete-severe":18,
    "dry-mud":19,
    "dry-gravel":20,
    "dry-asphalt-slight":21,
    "dry-asphalt-smooth":22,
    "dry-asphalt-severe":23,
    "dry-concrete-slight":24,
    "dry-concrete-smooth":25,
    "dry-concrete-severe":26,
}
info_set = create_subset_CSV(cartella_base, class_dictionary)
nome_file = './train.csv'
scrivi_su_csv(info_set["train"], nome_file)
nome_file = './test.csv'
scrivi_su_csv(info_set["test"], nome_file)
nome_file = './val.csv'
scrivi_su_csv(info_set["validation"], nome_file)
print("finitooo")
