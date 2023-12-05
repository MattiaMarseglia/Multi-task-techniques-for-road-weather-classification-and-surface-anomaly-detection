import os
import random
import shutil

def suddividi_immagini(cartella_principale, cartella_destinazione, prob_train=0.6, prob_validation=0.3, prob_test=0.1):
    # Crea le tre sotto-cartelle se non esistono già
    cartella_train = os.path.join(cartella_destinazione, "train")
    cartella_validation = os.path.join(cartella_destinazione, "validation")
    cartella_test = os.path.join(cartella_destinazione, "test")
    os.makedirs(cartella_train, exist_ok=True)
    os.makedirs(cartella_validation, exist_ok=True)
    os.makedirs(cartella_test, exist_ok=True)

    # Lista delle estensioni dei file delle immagini che desideri considerare
    estensioni_immagini = ['.jpg']

    # Scansiona tutte le cartelle nella cartella principale
    for cartella in os.listdir(cartella_principale):
        cartella_completa = os.path.join(cartella_principale, cartella)
        if os.path.isdir(cartella_completa):
            # Crea una lista di tutte le immagini nella cartella corrente
            immagini = [os.path.join(cartella_completa, file) for file in os.listdir(cartella_completa) if file.lower().endswith(tuple(estensioni_immagini))]
            
            # Genera un numero casuale tra 0 e 1
            probabilita = random.random()
            if probabilita < 0.6:
                for immagine in immagini:
                    shutil.move(immagine, os.path.join(cartella_train, os.path.basename(immagine)))
            elif probabilita < 0.9:
                for immagine in immagini:
                    shutil.move(immagine, os.path.join(cartella_validation, os.path.basename(immagine)))
            else:
                for immagine in immagini:
                    shutil.move(immagine, os.path.join(cartella_test, os.path.basename(immagine)))

# Definisci il percorso della cartella principale contenente le immagini
cartella_principale = "/mnt/sdc1/mmarseglia/dataset2/"

# Definisci il percorso della cartella destinazione delle suddivisioni
cartella_destinazione = "/mnt/sdc1/mmarseglia/dataset2_effective/"

# Imposta un seed per generare gli stessi valori casuali in modo coerente
seed = 42
random.seed(seed)

# Esegui la suddivisione delle immagini
suddividi_immagini(cartella_principale, cartella_destinazione)

print("Suddivisione completata.")
