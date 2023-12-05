import json

def count_json_class(json_path):
    # Leggi il file JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    # Dizionario per contare le categorie
    categorie_contate = {}

    # Scansiona gli elementi del file JSON
    for elemento in data.values():
        if 'categories' in elemento:
            categorie = elemento['categories']
            class_str = ""
            for index, categoria in enumerate(categorie):
                class_str += categoria
                if index < len(categorie)-1:
                    class_str += "-"
            if class_str in categorie_contate:
                categorie_contate[class_str] += 1
            else:
                categorie_contate[class_str] = 1
    return categorie_contate

json_path_val = '/user/mmarseglia/recursive_model_paper_folder/orderless-rnn-classification-master/RSCD_val.json'
json_path_train = '/user/mmarseglia/recursive_model_paper_folder/orderless-rnn-classification-master/RSCD_train.json'
categorie_contate_val = count_json_class(json_path_val)
categorie_contate_train = count_json_class(json_path_train)
# Stampare le categorie e il numero di volte che compaiono
print("categorie che non sono presenti nel val")
for categoria, conteggio in categorie_contate_train.items():
    if categoria not in categorie_contate_val.keys():
        print(f'{categoria}: {conteggio}')
# Stampare le categorie e il numero di volte che compaiono
print("categorie che non sono presenti nel train")
for categoria, conteggio in categorie_contate_val.items():
    if categoria not in categorie_contate_train.keys():
        print(f'{categoria}: {conteggio}')