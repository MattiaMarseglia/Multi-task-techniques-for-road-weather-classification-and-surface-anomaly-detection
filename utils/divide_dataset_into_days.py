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
import shutil

All_files = []

def verifica_e_crea_cartella(cartella):
    if not os.path.exists(cartella):
        os.makedirs(cartella)

def copia_immagine(origine, destinazione):
    if os.path.exists(origine):
        try:
            shutil.copy2(origine, destinazione)
        except:
            print('una copia non è avvenuta con successo:', origine)
            input()

def reorganize_dataset(cartella):
    i = 0
    for dirpath, dirnames, filenames in os.walk(cartella):
        print(i,", ",dirnames, len(filenames))
        i += 1
        for filename in filenames:
            starting_path = dirpath + '/' + filename
            # print(starting_path)
            date = filename[:4]+"-"+filename[4:6]+"-"+filename[6:8]
            path_to_save = "/home/mattia/Desktop/Tesi/dataset_reordered/" + date
            # print(path_to_save + '/' + filename)
            #"ciao da cami"
            verifica_e_crea_cartella(path_to_save)
            copia_immagine(starting_path, path_to_save + '/' + filename)


# Esempio di utilizzo:
cartella_base = "/home/mattia/Desktop/Tesi/RSCD dataset-1million"
reorganize_dataset(cartella_base)


