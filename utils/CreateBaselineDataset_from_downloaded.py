import os
import shutil

def sposta_immagine(origine, destinazione):
    if os.path.exists(origine):
        try:
            shutil.copy2(origine, destinazione)
        except:
            print('una copia non è avvenuta con successo:', origine)
            input()
    else:
        print("non trovato")

for dirpath, dirnames, filenames in os.walk("/home/mattia/Desktop/Tesi/RSCD dataset-1million/train"):
    
    for filename in filenames:
        if "jpg" in filename:
            print(dirpath+filename)
            destinazione = "/home/mattia/Desktop/Tesi/base_dataset/" + filename
            sposta_immagine(dirpath+"/"+filename, destinazione)