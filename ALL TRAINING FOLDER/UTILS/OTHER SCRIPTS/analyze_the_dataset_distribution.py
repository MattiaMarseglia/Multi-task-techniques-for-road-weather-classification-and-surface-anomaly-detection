import pandas as pd
import matplotlib.pyplot as plt

# Carica il training set e il validation set
training_set = pd.read_csv('/user/mmarseglia/baseline_csv/baseline_train.csv')
validation_set = pd.read_csv('/user/mmarseglia/baseline_csv/baseline_validation.csv')
test_set = pd.read_csv('/user/mmarseglia/baseline_csv/baseline_test.csv')

# Combina i tre dataset in uno unico
combined_data = pd.concat([training_set, validation_set, test_set])

# Calcola l'insieme di tutte le classi uniche nei tre set di dati
all_classes = set(combined_data['class'])

# Riorganizza le classi nei tre set di dati in base all'insieme di tutte le classi uniche
training_class_counts = training_set['class'].value_counts()
training_class_counts = training_class_counts.reindex(all_classes, fill_value=0)

validation_class_counts = validation_set['class'].value_counts()
validation_class_counts = validation_class_counts.reindex(all_classes, fill_value=0)

test_class_counts = test_set['class'].value_counts()
test_class_counts = test_class_counts.reindex(all_classes, fill_value=0)

# Crea un grafico a barre per la distribuzione delle classi
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
training_class_counts.plot(kind='bar', color='blue')
plt.title('Distribuzione delle classi nel Training Set')
plt.xlabel('Classe')
plt.ylabel('Numero di campioni')

plt.subplot(1, 3, 2)
validation_class_counts.plot(kind='bar', color='orange')
plt.title('Distribuzione delle classi nel Validation Set')
plt.xlabel('Classe')
plt.ylabel('Numero di campioni')

plt.subplot(1, 3, 3)
test_class_counts.plot(kind='bar', color='green')
plt.title('Distribuzione delle classi nel Test Set')
plt.xlabel('Classe')
plt.ylabel('Numero di campioni')

plt.tight_layout()

# Creazione dei DataFrame per il conteggio delle classi
training_class_counts_df = pd.DataFrame({'Classe': training_class_counts.index, 'Numero di campioni': training_class_counts.values})
validation_class_counts_df = pd.DataFrame({'Classe': validation_class_counts.index, 'Numero di campioni': validation_class_counts.values})
test_class_counts_df = pd.DataFrame({'Classe': test_class_counts.index, 'Numero di campioni': test_class_counts.values})

# Creazione del nome del file Excel
excel_file_name = 'distribuzione_classi.xlsx'

# Salvataggio dei DataFrame in un file Excel
with pd.ExcelWriter(excel_file_name) as writer:
    training_class_counts_df.to_excel(writer, sheet_name='Training Set', index=False)
    validation_class_counts_df.to_excel(writer, sheet_name='Validation Set', index=False)
    test_class_counts_df.to_excel(writer, sheet_name='Test Set', index=False)

print(f'Dati della distribuzione delle classi salvati in "{excel_file_name}"')


# Salva il plot come un'immagine
plt.savefig('distribuzione_classi_nuovaaaaaaaaaaaaaaa.png')
