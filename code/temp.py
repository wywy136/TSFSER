import os
import csv

file = open("./data/BPC_path2.csv", 'r')
csv_reader = csv.reader(file)
header = next(csv_reader)
for row in csv_reader:
    if not os.path.isfile(row[1]):
        print(row[1])
