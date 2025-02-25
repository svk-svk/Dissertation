import pandas as pd
import  time

start_time = time.time()

df = pd.read_csv(filepath_or_buffer="PicoDataSetKerberos.csv")

df.dropna(subset=['client'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['till'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['success'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.info()

df_anomaly = pd.read_csv(filepath_or_buffer="Anomaly_Set.csv")
df_anomaly.info()
list_uid_anomaly=df_anomaly["uid"].tolist()
print(list_uid_anomaly)

df_training= pd.read_csv(filepath_or_buffer="2019-07-19.csv")
df_training.dropna(subset=['client'], inplace=True)
df_training.reset_index(drop=True, inplace=True)

df_training.dropna(subset=['till'], inplace=True)
df_training.reset_index(drop=True, inplace=True)

df_training.dropna(subset=['success'], inplace=True)
df_training.reset_index(drop=True, inplace=True)
df_training.info()

df_training.to_csv('2019-07-19_clean.csv', index=False)

unique_counts = df_training['success'].value_counts()
print("\nUnique Value Counts:")
print(unique_counts)


tuple_list = {(row['id.orig_h'], row['client'], row['service']) for index, row in df_training.iterrows()}

traning_tuple = list(tuple_list)
print(len(traning_tuple))

printed_tuples = set()
uid_identified_as_anomalies=[]


for index, row in df.iterrows():
    current_tuple = (row['id.orig_h'], row['client'], row['service'])


    if current_tuple not in traning_tuple and current_tuple not in printed_tuples: #78-8 pentru prima aparitie 183-10 pentru toate aparitiile se face cu lista

        uid_identified_as_anomalies.append(row["uid"])

        printed_tuples.add(current_tuple)


for y in printed_tuples:
    print(y)
print(len(printed_tuples))

uid_true_positive_anomalies = [item for item in uid_identified_as_anomalies if item in list_uid_anomaly]


total_lenght_dataset=len(df)

true_positives=len(uid_true_positive_anomalies)#identificatele bune
false_positives=len(uid_identified_as_anomalies)-true_positives #identificatele rau
false_negatives=len(list_uid_anomaly)-true_positives #neidentificatele
true_negatives= total_lenght_dataset - len(list_uid_anomaly) - false_positives

score = len(uid_true_positive_anomalies)
lateral_movement_detection_rate = score / len(uid_identified_as_anomalies)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
fpr = false_positives/(false_positives+true_negatives)
f_score = 2 * ((precision * recall) / (precision + recall))

end_time = time.time()
total_time = end_time - start_time

print(f"{precision} precision")
print(f"{recall} recall")
print(f"{fpr} fpr")
print(f"{f_score} f1_score")
print(f"{len(uid_identified_as_anomalies)} total anomalies find")
print(f"{score} total corect anomalies")


print(f"Execution time: {total_time:.6f} seconds")


