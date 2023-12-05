# !/usr/bin/python
########## FILE NAMES ##########
### TRAIN LABEL ###
#   202201261342122-dry-asphalt-severe
### TEST ###
#   202201252342299-dry-asphalt-smooth 
### VALIDATION ###
#   202205171728-dry-concrete-smooth
import os
import csv

All_files = []

def leggi_nomi_file(cartella):
    i = 0
    for dirpath, dirnames, filenames in os.walk(cartella_base):
        print(i,", ",dirnames, len(filenames))
        i += 1
        for filename in filenames:
            All_files.append([filename,filename[:4]+"-"+filename[4:6]+"-"+filename[6:8]]) 
    return All_files


# Esempio di utilizzo:
cartella_base = "/home/mattia/Desktop/Tesi/RSCD dataset-1million"
All_files = leggi_nomi_file(cartella_base)

All_sorted_files = sorted(All_files, key=lambda x: x[0])

classes = {"class":{}}
current_date = All_sorted_files[0][1]
num_images = 0
for index, row in enumerate(All_sorted_files):
    print((index/len(All_sorted_files))*100,"%")
    if row[1] != current_date:
        current_date = row[1]
    sub_strings = row[0].split("-")
    if str(sub_strings[1:]) not in classes["class"].keys():
        classes["class"][str(sub_strings[1:])] =  [0, current_date]
    else:
        if current_date not in classes["class"][str(sub_strings[1:])][1].split(","):
            classes["class"][str(sub_strings[1:])][1] += "," + current_date
        classes["class"][str(sub_strings[1:])][0] += 1
    num_images += 1
filename = '/home/mattia/Desktop/Tesi/info.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for c, v in classes["class"].items():
        writer.writerow([c, str(v)])
print(num_images)
########### UNCOMMENT TO CREATE THE CSV image name, ordered date date ###########
# filename = '/home/mattia/Desktop/Tesi/data.csv'
# with open(filename, 'w', newline='') as csvfile:
    # writer = csv.writer(csvfile)
    # for item in All_sorted_files:
        # writer.writerow([item,item[:4]+"-"+item[4:6]+"-"+item[6:8]])

# print(len(All_files))
# print(All_files[0])
# print(All_files[-1])

