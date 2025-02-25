import os, csv, warnings, time
import pandas as pd

from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from module import ipv4_address_to_int, bool_to_int,  split_row_client, print_column_info, process_column
from multiprocessing import Manager


warnings.simplefilter("ignore", category=FutureWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df=pd.read_csv(filepath_or_buffer="PicoDataSetKerberos.csv")
#aservice 1206 si client 1097
df.dropna(subset=['client'], inplace=True)
df.info()
df.reset_index(drop=True, inplace=True)
df.info()

#df["uid"] #este unic pentru fiecare log
#Excutand functia count_apparances pe coloana "id.resp_h" ne returneaza lista cu un singur element si anume [10.99.99.5] reprezentand adresa serverului pe care ruleaza protocolul kerberos din asta rezulta ca nu avem mai mult servere de kerberos coloana aceasta nu aduce informatii suplimentare <=>  nu vom folosi coloana
df = df.drop(columns=['id.resp_h'])

# Dupa ce stergem liniile dupa existeanta sa nu a service verificam id.resp_p cu functia appearences vedem ca ne returneaza doar lista cu elementrul 88 [88] => ca coloana id.resp_p nu aduce nici o informatie utila <=> nu o vom folosi
df = df.drop(columns=['id.resp_p'])

df['ts']= pd.to_datetime(df['ts']).dt.tz_localize(None)
df['ts'] = df['ts'].astype('int64')##pt isforest

df['id.orig_h'] = df['id.orig_h'].apply(ipv4_address_to_int)
print_column_info(df, 'id.orig_h')


df.dropna(subset=['till'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['till']= pd.to_datetime(df['till']).dt.tz_localize(None)
df['till'] = df['till'].astype('int64') ####pt isforest
print_column_info(df, 'till')

df.dropna(subset=['success'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['success'] = df['success'].apply(bool_to_int)
print_column_info(df, 'success')

df['forwardable']= df['forwardable'].apply(bool_to_int)
print_column_info(df, 'forwardable')

df['renewable']=df['renewable'].apply(bool_to_int)
print_column_info(df, 'renewable')

process_column(df, "request_type")
process_column(df, "cipher")
process_column(df, "error_msg")
process_column(df, "service")


df[['clientstation', 'clientrealm']] = df['client'].apply(lambda x: split_row_client(x))
process_column(df, "clientstation")
process_column(df, "clientrealm")
process_column(df, "client")


df_anomaly=pd.read_csv(filepath_or_buffer="Anomaly_Set.csv")
print(df_anomaly)
df_anomaly.info()
list_uid_anomaly = list(df_anomaly["uid"])
print(list_uid_anomaly)

df.info()

best_score = -1

def process_parameters(n_estimators, max_samples, contamination, max_features, bootstrap,random_states, df, list_of_features,features_org,
                       list_uid_anomaly,lock):



    features = df[list_of_features]
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_states,
        n_jobs=-3
    )

    iso_forest.fit(features)
    df['iforest'] = pd.Series(iso_forest.predict(features))
    df['iforest'] = df['iforest'].map({1: 0, -1: 1})

    unique_counts = len(df["uid"].value_counts())
    list_uid_isolation = df["uid"][df['iforest'] == 1].tolist()


    uid_identified_as_anomalies = [item for item in list_uid_isolation if item in list_uid_anomaly]
    score = len(uid_identified_as_anomalies)

    if score > 3:

        total_lenght_dataset = len(df)

        true_positive = len(uid_identified_as_anomalies)  # identificatele bune
        false_positive = len(list_uid_isolation) - true_positive  # identificatele rau
        false_negative = len(list_uid_anomaly) - true_positive  # neidentificatele
        true_negative = total_lenght_dataset - len(list_uid_anomaly) - false_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        f_score = 2 * ((precision * recall) / (precision + recall))

        # print(f"{precision} precision")
        # print(f"{recall} recall")
        # print(f"{fpr} fpr")
        # print(f"{f_score} f1_score")
        # print(f"{len(uid_identified_as_anomalies)} total anomalies find")
        # print(f"{score} total corect anomalies")


        list_row=[len(list_uid_isolation) / score, len(list_uid_isolation),score,unique_counts,precision,recall,fpr, f_score,n_estimators,max_samples,contamination,max_features,bootstrap,random_states]
        list01_features=[1 if item in list_of_features else 0 for item in features_org]

        for el01 in list01_features:
            list_row.append(el01)
        list01_an=[1 if item in uid_identified_as_anomalies else 0 for item in  list_uid_anomaly]

        for el01 in list01_an:
            list_row.append(el01)
        with lock:
            with open(file="ISforestResults.csv", mode='a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list_row)

        result = {
            "score": score,
            "uid_if_nou_gasite": len(list_uid_isolation),
            "ratio": len(list_uid_isolation) / score,
            "uid_anomaly": len(list_uid_anomaly),
            "total_nr_valori_din_set": unique_counts,
            "params": {
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'contamination': contamination,
                'max_features': max_features,
                'bootstrap': bootstrap
            },
            "uid_identified_as_anomalies": uid_identified_as_anomalies
        }
    else:
        result = {
            "score": 0,
            "params": {
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'contamination': contamination,
                'max_features': max_features,
                'bootstrap': bootstrap
            },
            "uid_identified_as_anomalies": []
        }

    return result



if __name__ == '__main__':
    #param_grid = {'n_estimators': [7,10,30,50,9,75,100,300,200,50,350], 'max_samples': [1.0,0.4, 0.5, 0.3,0.2, 0.6,  0.75, 0.8,0.9,'auto'], 'contamination': [0.0092, 0.02,0.03,0.035,0.04,0.05, 0.06,0.07, 0.08], 'max_features': [0.5, 1.0, 0.75, 0.5, 0.9,0.4,  0.8, 0.6], 'bootstrap': [True,False]}
    # Example parameter grid
    # param_grid_unu = {
    #     'n_estimators': [7, 10, 30, 50, 9, 75, 100, 300, 200, 50, 350],
    #     'max_samples': ['auto', 1.0, 0.4, 0.5, 0.3, 0.2, 0.6, 0.75, 0.8, 0.9],
    #     'contamination': [0.0092, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08],
    #     'max_features': [0.5, 1.0, 0.75, 0.5, 0.9, 0.4, 0.8, 0.6],
    #     'bootstrap': [False, True]
    #
    # }

    # param_grid = {
    #     'n_estimators': [3,4,5,6,7,8,9, 10,11,12,15,30,20,25,50,45,40],
    #     'max_samples': ['auto', 1.0, 0.4, 0.5, 0.3, 0.2, 0.6, 0.75, 0.8, 0.9],
    #     'contamination': [0.008,0.007, 0.0092,0.015, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08,0.09,0.1,0.11],
    #     'max_features': [0.5, 1.0, 0.75, 0.5, 0.9, 0.4, 0.8, 0.6],
    #     'bootstrap': [False, True],
    #     'random_states':[42,2,100,1000]
    # }
    ##best
    param_grid = {
        'n_estimators': [3,4],
        'max_samples': [ 1.0, 0.5, 0.3, 0.8, 0.9],
        'contamination': [0.008, 0.015,  0.03,  0.04, 0.05,  0.08],
        'max_features': [1.0, 0.9],
        'bootstrap': [False, True],
        'random_states': [42, 2, 100]
    }

    features = ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service',
           'success', 'till', 'cipher', 'forwardable', 'renewable','error_msg', 'client','clientstation', 'clientrealm']

    #list_features = generate_combinations_features(features)

    # list_features = [
    #     ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service','success', 'till', 'cipher', 'forwardable', 'renewable','error_msg', 'client','clientstation',
    #      'clientrealm'],
    #     ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'forwardable',
    #      'renewable', 'error_msg',  'clientstation', 'clientrealm'],
    #     ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'error_msg', 'client'],
    #     ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'error_msg', 'client'],
    #     ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'error_msg', 'clientstation', 'clientrealm'],
    #     [ 'id.orig_h',  'service', 'success', 'error_msg', 'client']]

    list_features = [

        ['id.orig_h', 'service', 'success', 'error_msg', 'client'],

    ]


    param_combinations = [(n_estimators, max_samples, contamination, max_features, bootstrap,random_states)
                          for n_estimators in param_grid['n_estimators']
                          for max_samples in param_grid['max_samples']
                          for contamination in param_grid['contamination']
                          for max_features in param_grid['max_features']
                          for bootstrap in param_grid['bootstrap']
                          for random_states in param_grid['random_states']]

    header = ['ratio', 'uid_if_nou_gasite', 'score', 'total_nr_valori_din_set','precision','recall','fpr','f1_score', 'n_estimators', 'max_samples',
              'contamination', 'max_features', 'bootstrap', 'random_states',
              'ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service',
              'success', 'till', 'cipher', 'forwardable', 'renewable', 'error_msg', 'client', 'clientstation',
              'clientrealm',
              'CG2fHGatKXKTpsyh6', 'CRUEAq4YtJkoBSShid', 'CTk8QH20I5oC7wqQN9', 'C5XKvBZyCZ4Bpic65', 'CvE9GVJpVxLGrTsO2',
              'CJQGFI1WYwA4hnajS4', 'Cdo3iN17MkD2D4SSKk', 'CaKSyP3tBxtPmUfbs1', 'C1jGntzIYAIF8NB33',
              'C9qYNL1PMeaOlhux19']

    start_time = time.time()

    csvfile = open(file="ISforestResults.csv", mode='w+', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(header)
    csvfile.close()

    manager = Manager()
    lock = manager.Lock()

    results = Parallel(n_jobs=-3
                       )(

            delayed(process_parameters)(n_estimators, max_samples, contamination, max_features, bootstrap,random_states, df, features_item,features,
                                        list_uid_anomaly,lock)
            for (n_estimators, max_samples, contamination, max_features, bootstrap,random_states) in param_combinations
            for features_item in list_features

    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time: {total_time:.6f} seconds")
    csvfile.close()