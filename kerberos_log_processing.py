import os, fnmatch, csv
import pandas as pd

def find_paths(pattern, path):
    files_names = []
    directories_names=[]

    for root, dirs, files in os.walk(path):

        if root==path:
            continue
        else:
            directories_names.append( str(root).replace(path+"\\",""))
            aux=[]

        for name in files:

            if fnmatch.fnmatch(name, pattern):

                aux.append(os.path.join(root, name))
        files_names.append(aux)

    return files_names, directories_names

list_labels = []

files, directories=find_paths('kerberos.*.log', 'E:/Disertatie/PicoDomain/Zeek_Logs')
All_data = [["Header"]]
for x in files:

    temporary_data_days=[]

    for y in x:
        y=y.replace("\\","/")
        csv_name=y.replace("log","csv")

        with open(file=y, mode= 'r') as logfile:
            logreader = csv.reader(logfile, delimiter='\t')
            data = []
            for row in logreader:
                row_string=row[0]

                row_string = row_string.replace("{", "")
                row_string = row_string.replace("}", "")
                row_string = row_string.split(',')

                my_dict = {}

                for pair in row_string:
                    key, value = pair.split('":')
                    key=key.replace('"', "")
                    value=value.replace('"', "")
                    my_dict[key] = value
                    if key not in list_labels:
                        list_labels.append(key)

                data.append(my_dict)


        with open(file=csv_name, mode='w+') as csvfile:
            number_rows=0
            csvwriter = csv.writer(csvfile)
            All_data[0]=list_labels
            csvwriter.writerow(list_labels)
            number_rows=number_rows+1
            #for all logs
            # for line in data:
            #     newlist=[]
            #     for label in list_labels:
            #         if label in line:
            #             newlist.append(line[label])
            #
            #         else:
            #             newlist.append(None)
            #
            #     csvwriter.writerow(newlist)
            #     temporary_data_days.append(newlist)
            #     number_rows = number_rows + 1

            #for logs that have more then 6 features
            for line in data:
                if line.__len__() == 6:
                    continue
                newlist=[]
                for label in list_labels:
                    if label in line:
                        newlist.append(line[label])

                    else:
                        newlist.append(None)

                csvwriter.writerow(newlist)
                temporary_data_days.append(newlist)
                number_rows = number_rows + 1
    All_data.append(temporary_data_days)

with open("PicoDataSetKerberos.csv", 'w', newline='') as all:
#with open("PicoDataSetKerberos_ALL.csv", 'w', newline='') as all:
    writer_all = csv.writer(all)
    header = All_data[0]
    writer_all.writerow(header)

    for folder_name in directories:
        position= directories.index(folder_name)


        with open(folder_name+".csv", 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(header)

            writer.writerows(All_data[position+1])
            writer_all.writerows(All_data[position + 1])

excel_file = 'Anomaly_Set.xlsx'


df = pd.read_excel(excel_file)

csv_file = 'Anomaly_Set.csv'
df.to_csv(csv_file, index=False)

print(f"Excel file {excel_file} has been successfully converted to CSV file {csv_file}")