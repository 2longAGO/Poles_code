import csv

with open('CheckPoints.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    with open('Half_CheckPoints_.csv', 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        for row in csv_reader:
            if line_count%2 == 0:
                csv_writer.writerow(row)
            line_count+=1
    print(f'Processed {line_count} lines.') # 92 180 positions 46 090

