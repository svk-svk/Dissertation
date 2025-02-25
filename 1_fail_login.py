import pandas as pd
import  time

start_time = time.time()

df= pd.read_csv(filepath_or_buffer="PicoDataSetKerberos.csv")
#service 1206 si client 1097 : KDC_ERR_BADOPTION 91 + KDC_ERR_S_PRINCIPAL_UNKNOWN 12 +KRB_AP_ERR_TKT_EXPIRED 2+4 nan

df.dropna(subset=['client'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['till'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['success'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.info()

unique_counts = df['success'].value_counts()
print("\nUnique Value Counts:")
print(unique_counts)


uid_identified_as_anomalies =df["uid"][df['success']==False].tolist()

print( uid_identified_as_anomalies)
df_anomaly = pd.read_csv(filepath_or_buffer="Anomaly_Set.csv")

list_uid_anomaly=df_anomaly["uid"].tolist()
print(list_uid_anomaly)

uid_true_positive_anomalies = [item for item in uid_identified_as_anomalies if item in list_uid_anomaly]
total_lenght_dataset=len(df)

true_positives=len(uid_true_positive_anomalies)
false_positives=len(uid_identified_as_anomalies)-true_positives
false_negatives=len(list_uid_anomaly)-true_positives
true_negatives= total_lenght_dataset- len(list_uid_anomaly)- false_positives

score = len(uid_true_positive_anomalies)
lateral_movement_detection_rate= score / len(uid_identified_as_anomalies)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
fpr= false_positives/(false_positives+true_negatives)
f_score = 2 * ((precision * recall) / (precision + recall))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"{precision} precision")
print(f"{recall} recall")
print(f"{fpr} fpr")
print(f"{f_score} f1_score")
print(f"{len(uid_identified_as_anomalies)} total anomalies find")
print(f"{score} total corect anomalies")
#print(f"{len(uid_identified_as_anomalies)/score} raport anomalii gasite anomalii corecte")



print(f"Elapsed time: {elapsed_time:.6f} seconds")