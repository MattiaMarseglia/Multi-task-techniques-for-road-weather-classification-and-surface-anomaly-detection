import pandas as pd

# Leggi i tre file CSV in tre DataFrame separati
file1 = pd.read_csv("/user/mmarseglia/second_split_60_30_10/test_dataset2_Multihead.csv")
file2 = pd.read_csv("/user/mmarseglia/second_split_60_30_10/train_dataset2_Multihead.csv")
file3 = pd.read_csv("/user/mmarseglia/second_split_60_30_10/validation_dataset2_Multihead.csv")

# Creare la colonna 'image_prefix' in tutti e tre i DataFrame
file1['image_prefix'] = file1['image'].str[:8]
file2['image_prefix'] = file2['image'].str[:8]
file3['image_prefix'] = file3['image'].str[:8]

# Ordina i DataFrame in base alla colonna 'image_prefix'
file1.sort_values(by='image_prefix', inplace=True)
file2.sort_values(by='image_prefix', inplace=True)
file3.sort_values(by='image_prefix', inplace=True)

# Verifica se ci sono elementi comuni nella colonna 'image_prefix' tra i tre DataFrame
duplicates_across_files1 = file1[file1['image_prefix'].isin(file2['image_prefix']) | file1['image_prefix'].isin(file3['image_prefix'])]
duplicates_across_files2 = file2[file2['image_prefix'].isin(file1['image_prefix']) | file2['image_prefix'].isin(file3['image_prefix'])]
duplicates_across_files3 = file3[file3['image_prefix'].isin(file2['image_prefix']) | file3['image_prefix'].isin(file1['image_prefix'])]
prova1 = file1[file1['image_prefix'].isin(file1['image_prefix'])]
prova2 = file2[file2['image_prefix'].isin(file2['image_prefix'])]
prova3 = file3[file3['image_prefix'].isin(file3['image_prefix'])]

# Stampa i duplicati trovati
print(duplicates_across_files1)
print(duplicates_across_files2)
print(duplicates_across_files3)
print(len(prova1)+len(prova2) + len(prova3))
