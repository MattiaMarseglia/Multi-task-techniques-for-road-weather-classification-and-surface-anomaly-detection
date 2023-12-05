import csv
import os

def analyze_csv_files(folder_path, separator):
    data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(folder_path, filename)

            with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip the header row

                for row in csv_reader:
                    if len(row) >= 3:
                        values = row[2].split(separator)
                        for value in values:
                            value = value.strip()  # Remove leading/trailing spaces
                            if value in data_dict:
                                data_dict[value] += 1
                            else:
                                data_dict[value] = 1

    return data_dict

def save_dict_to_file(data_dict, output_file):
    sorted_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))

    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in sorted_dict.items():
            file.write("'" + key + "', ")

if __name__ == "__main__":
    folder_path = "/mnt/sdc1/mmarseglia/dataset"
    output_file = "frequency_class_in_dataset.txt"
    separator = "-"  # Separator for splitting the third column

    result_dict = analyze_csv_files(folder_path, separator)
    save_dict_to_file(result_dict, output_file)

    print("Analysis completed and dictionary saved to", output_file)
