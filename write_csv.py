import csv

w20 = csv.writer(open('path/xxx.csv', "a"))

for k, v in data_record.items():
    w20.writerow([k, v])
