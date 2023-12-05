import csv

# Definisci il nome del file CSV
file_csv = '/user/mmarseglia/train_reaordered_single_head.csv'

# Crea un dizionario vuoto per memorizzare le associazioni tra "class" e "label"
associazioni = {}

# Apre il file CSV in modalità di lettura
with open(file_csv, 'r') as csv_file:
    # Legge il file CSV con il modulo csv di Python
    csv_reader = csv.DictReader(csv_file)
    
    # Itera attraverso le righe del file CSV
    for row in csv_reader:
        # Estrai i valori delle colonne "class" e "label" dalla riga corrente
        classe = row['class']
        label = int(row['label'])
        
        # Aggiungi l'associazione al dizionario
        associazioni[classe] = label

# Stampare il dizionario risultante
print(associazioni)
