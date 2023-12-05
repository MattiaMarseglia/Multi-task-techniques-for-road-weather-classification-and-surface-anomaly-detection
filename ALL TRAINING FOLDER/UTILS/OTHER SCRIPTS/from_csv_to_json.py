import csv
import json
import os

# Inizializza un dizionario vuoto per salvare i dati formattati
dati_formattati = {}

# Specifica il percorso del file CSV da leggere
percorso_file_csv = '/user/mmarseglia/second_split_60_30_10/test_dataset2_Multihead.csv'  # Ricorda di sostituire con il percorso reale del tuo file CSV

# Apre il file CSV e legge le righe, saltando la prima riga
with open(percorso_file_csv, 'r') as file_csv:
    lettore_csv = csv.reader(file_csv)
    
    # Salta la prima riga
    next(lettore_csv)
    
    # Itera attraverso le righe del file CSV
    for riga in lettore_csv:
        # Assicurati che la riga abbia almeno tre elementi prima di tentare di accedere agli elementi
        if len(riga) >= 3:
            chiave = riga[0]  # Primo elemento come chiave
            url = riga[1]  # Secondo elemento come URL
            categorie = riga[2].split("-")  # Terzo e successivi elementi come categorie
            
            dati_formattati[chiave] = {
                "url": url,
                "categories": categorie
            }

# Nome del file JSON da creare (stesso nome del CSV ma con estensione .json)
nome_file_json = os.path.splitext(os.path.basename(percorso_file_csv))[0] + '.json'

# Salva il dizionario dei dati formattati in un file JSON con indentazione per la leggibilità
with open(nome_file_json, 'w') as file_json:
    json.dump(dati_formattati, file_json, indent=4)

print(f"Dati formattati salvati in {nome_file_json}")
