import os
import shutil

# Specifica la cartella di origine (dove si trovano i file da copiare)
cartella_origine = '/mnt/sdc1/mmarseglia/dataset/test'  # Sostituisci con il percorso della tua cartella di origine

# Specifica la cartella di destinazione (dove desideri copiare i file)
cartella_destinazione = '/mnt/sdc1/mmarseglia/dataset/dataset'  # Sostituisci con il percorso della tua cartella di destinazione

# Verifica se la cartella di origine esiste
if not os.path.exists(cartella_origine):
    print(f"La cartella di origine '{cartella_origine}' non esiste.")
else:
    # Verifica se la cartella di destinazione esiste, altrimenti creala
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    # Ottieni la lista di tutti i file nella cartella di origine
    elenco_file = os.listdir(cartella_origine)

    # Loop attraverso tutti i file nella cartella di origine e copiali nella cartella di destinazione
    for file in elenco_file:
        percorso_file_origine = os.path.join(cartella_origine, file)
        percorso_file_destinazione = os.path.join(cartella_destinazione, file)
        shutil.copy2(percorso_file_origine, percorso_file_destinazione)
        print(f"Copiato '{file}' in '{cartella_destinazione}'")
    
    print("Copia completata.")
