import csv

def replace_second_element_with_multi(file_path, class_dictionary):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)
        header = rows[0]  # Memorizza l'intestazione
        updated_rows = [header]

        for row in rows[1:]:  # Salta la prima riga (intestazione)
            row[1] = class_dictionary[str(row[1])]  # Sostituisci il secondo elemento con la tripla
            updated_rows.append(row)

    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(updated_rows)

if __name__ == "__main__":

    #DICTIONARY FROM MONOCLASS TO MULTICLASS 
    class_dictionary = {
        "0":"3-5-4",
        "1":"3-3-4",
        "2":"3-4-4",
        "3":"3-1-2",
        "4":"3-1-3",
        "5":"1-1-0",
        "6":"0-1-0",
        "7":"2-1-0",
        "8":"1-1-1",
        "9":"0-1-1",
        "10":"2-1-1",
        "11":"3-2-2",
        "12":"3-2-3",
        "13":"1-2-0",
        "14":"0-2-0",
        "15":"2-2-0",
        "16":"1-2-1",
        "17":"0-2-1",
        "18":"2-2-1",
        "19":"3-0-2",
        "20":"3-0-3",
        "21":"1-0-0",
        "22":"0-0-0",
        "23":"2-0-0",
        "24":"1-0-1",
        "25":"0-0-1",
        "26":"2-0-1",
    }

    csv_file_path = "/home/mattia/Desktop/Tesi/datasets/dataset_reordered/val copy.csv"
    replace_second_element_with_multi(csv_file_path, class_dictionary)
