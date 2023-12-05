import os
import csv

# Specifica la cartella contenente i file immagine
cartella = '/mnt/sdc1/mmarseglia/dataset2_effective/test'

# Dizionario di mappatura delle classi
mappa_classi = {
    'dry-concrete-severe': 26,
    'ice': 0,
    'dry-asphalt-smooth': 22,
    'wet-concrete-smooth': 9,
    'dry-mud': 19,
    'dry-gravel': 20,
    'wet-mud': 3,
    'wet-gravel': 4,
    'water-asphalt-smooth': 14,
    'dry-concrete-smooth': 25,
    'water-concrete-smooth': 17,
    'wet-asphalt-smooth': 6,
    'fresh_snow': 1,
    'wet-asphalt-slight': 5,
    'dry-concrete-slight': 24,
    'dry-asphalt-slight': 21,
    'wet-concrete-severe': 10,
    'melted_snow': 2,
    'water-concrete-severe': 18,
    'water-mud': 11,
    'water-asphalt-severe': 15,
    'water-asphalt-slight': 13,
    'water-gravel': 12,
    'wet-asphalt-severe': 7,
    'wet-concrete-slight': 8,
    'dry-asphalt-severe': 23,
    'water-concrete-slight': 16
 }

# Lista per memorizzare le righe del CSV
righe_csv = []

# Scansiona la cartella e crea le righe del CSV
for nome_file in os.listdir(cartella):
    if nome_file.endswith('.jpg'):
        # Estrai il nome della classe dalla parte iniziale del nome del file
        nome_classe = nome_file.replace(nome_file.split('-')[0]+'-', "")
        nome_classe = nome_classe.replace(".jpg", "")
        classe = mappa_classi.get(nome_classe, 'ClasseSconosciuta')  # Usa 'ClasseSconosciuta' se la classe non è nel dizionario

        # Aggiungi la riga al CSV
        riga = f'{nome_file},{classe},{nome_classe}'
        righe_csv.append(riga)

# Scrivi le righe nel file CSV
with open('test_dataset2_single_head.csv', 'w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    writer.writerow(['image', 'label', 'class'])  # Intestazione del CSV
    writer.writerows([riga.split(',') for riga in righe_csv])

print("File CSV creato con successo.")
