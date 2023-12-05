import os
import json
import csv
import debugpy

# Funzione per scorrere ricorsivamente la cartella e trovare i file JSON
def trova_file_json(directory):
    file_json = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_json.append(os.path.join(root, file))
    return file_json

def json_to_lista_dizionari(file_json):
    dati = []
    for file in file_json:
        with open(file, 'r') as json_file:
            dati_file = json.load(json_file)
            dati_file = {'file_path': os.path.basename(file).split(".")[0], **dati_file} 
            try:
                dati_file['trainData'] = {"TL": [item['TrainLoss'] for item in dati_file['trainData']], "TA": [item['Valacc'] for item in dati_file['trainData']]} 
                dati_file['valData'] = {"VL": [item['ValLoss'] for item in dati_file['valData']], "VA": [item['Valacc'] for item in dati_file['valData']]} 
                dati.append(dati_file)
            except:
                pass
    # Ordina la lista di dizionari in base alla chiave 'file_path'
    dati_ordered = sorted(dati, key=lambda x: x['file_path'])
    return dati_ordered


# Funzione per scrivere i dati in un file CSV
def scrivi_su_csv(dati, nome_file_csv):
    with open(nome_file_csv, 'w', newline='') as csv_file:
        campo_nomi = dati[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=campo_nomi)
        writer.writeheader()
        for riga in dati:
            writer.writerow(riga)

try:
    debugpy.listen(("localhost", 5679))
    debugpy.wait_for_client()
except:
    print("non mi fermo")
    pass

# Directory da cui iniziare la ricerca dei file JSON
directory = '/user/mmarseglia/first_split_as_paper'

# Trova i file JSON nella cartella e nelle sottocartelle
file_json = trova_file_json(directory)

# Converti i file JSON in una lista di dizionari
dati = json_to_lista_dizionari(file_json)

# Scrivi i dati nel file CSV
nome_file_csv = 'all_training_view.csv'
scrivi_su_csv(dati, nome_file_csv)

print(f'Dati scritti in {nome_file_csv}')
