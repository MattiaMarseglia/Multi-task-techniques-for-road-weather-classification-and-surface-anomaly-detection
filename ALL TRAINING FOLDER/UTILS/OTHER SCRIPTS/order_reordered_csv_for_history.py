import pandas as pd


def order(start_file, output_file):
    # Leggi il file CSV
    df = pd.read_csv(start_file)

    # Estrai il primo numero prima del trattino nella colonna "label"
    df['first_number'] = df['image'].str.split('-').str[0]

    # Ordina il DataFrame in base al nuovo campo "first_number"
    df = df.sort_values(by='first_number')

    # Elimina il campo temporaneo "first_number" se non lo desideri nel risultato finale
    df = df.drop(columns=['first_number'])

    # Salva il DataFrame ordinato in un nuovo file CSV
    df.to_csv(output_file, index=False)



start_file = "/user/mmarseglia/second_split_60_30_10/test_dataset2_multihead.csv"
output_file = '/user/mmarseglia/second_split_60_30_10/test_dataset2_multihead_ordered.csv'
order(start_file, output_file)
start_file = "/user/mmarseglia/second_split_60_30_10/train_dataset2_multihead.csv"
output_file = '/user/mmarseglia/second_split_60_30_10/train_dataset2_multihead_ordered.csv'
order(start_file, output_file)
start_file = "/user/mmarseglia/second_split_60_30_10/validation_dataset2_multihead.csv"
output_file = '/user/mmarseglia/second_split_60_30_10/validation_dataset2_multihead_ordered.csv'
order(start_file, output_file)