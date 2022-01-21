import csv

csv_data = open('./data/BPC_path1.csv')
ori_data = []
csv_reader = csv.reader(csv_data)
header = next(csv_reader)
# [0,/project/graziul/data/Zone1/2018_08_12/201808120932-28710-27730.mp3,00.02.21.252,00.02.31.279,RADIOSHOP TESTING ONE TWO THREE FOUR FIVE FIVE FOUR THREE TWO ONE RADIO SHOP TEST,10.027]
for row in csv_reader:
    ori_data.append(row)
    
    start, end = row[2], row[3]
    
    s, e = start.split('.'), end.split('.')
    if len(s) != 4:
        print(start)
    if len(e) != 4:
        print(end)