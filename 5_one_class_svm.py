from joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import  csv,  time
import pandas as pd
from multiprocessing import Manager

from module import ipv4_address_to_int, bool_to_int, generate_combinations_features, count_appearances, list_to_nan_first_position_list, to_int_token_from_list, split_row_client, print_column_info, process_column



#def process_parameters_ocsvm(kernel, gamma, nu, df, list_of_features, features_org, list_uid_anomaly, lock):
def process_parameters_ocsvm(kernel, degree, gamma, coef0, tol, nu, shrinking, cache_size, max_iter, df, list_of_features, features_org, list_uid_anomaly, lock):
    features = df[list_of_features]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    #ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu, shrinking=shrinking,
                        cache_size=cache_size, max_iter=max_iter)
    ocsvm.fit(features_scaled)
    df['ocsvm'] = pd.Series(ocsvm.predict(features_scaled))
    df['ocsvm'] = df['ocsvm'].map({1: 0, -1: 1})

    unique_counts = len(df["uid"].value_counts())
    list_uid_ocsvm = df["uid"][df['ocsvm'] == 1].tolist()

    uid_identified_as_anomalies = [item for item in list_uid_ocsvm if item in list_uid_anomaly]
    score = len(uid_identified_as_anomalies)

    if score > 3:
        total_length_dataset = len(df)
        true_positive = len(uid_identified_as_anomalies)
        false_positive = len(list_uid_ocsvm) - true_positive
        false_negative = len(list_uid_anomaly) - true_positive
        true_negative = total_length_dataset - len(list_uid_anomaly) - false_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        f_score = 2 * ((precision * recall) / (precision + recall))

        print(f"{precision} precision")
        print(f"{recall} recall")
        print(f"{fpr} fpr")
        print(f"{f_score} f1_score")
        print(f"{len(uid_identified_as_anomalies)} total anomalies find")
        print(f"{score} total correct anomalies")
        print(f"{len(uid_identified_as_anomalies) / score} ratio of found anomalies to correct anomalies")

        list_row = [len(list_uid_ocsvm) / score, len(list_uid_ocsvm), score, unique_counts, precision, recall, fpr,
                    f_score, kernel, gamma, nu]

        list01_features = [1 if item in list_of_features else 0 for item in features_org]

        for el01 in list01_features:
            list_row.append(el01)
        list01_an = [1 if item in uid_identified_as_anomalies else 0 for item in list_uid_anomaly]

        for el01 in list01_an:
            list_row.append(el01)

        with lock:
            with open(file="OCSVMResults.csv", mode='a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list_row)

        result = {
            "score": score,
            "uid_if_nou_gasite": len(list_uid_ocsvm),
            "ratio": len(list_uid_ocsvm) / score,
            "uid_anomaly": len(list_uid_anomaly),
            "total_nr_valori_din_set": unique_counts,
            "params": {
                'kernel': kernel,
                'gamma': gamma,
                'nu': nu
            },
            "uid_identified_as_anomalies": uid_identified_as_anomalies
        }
    else:
        result = {
            "score": 0,
            "params": {
                'kernel': kernel,
                'gamma': gamma,
                'nu': nu
            },
            "uid_identified_as_anomalies": []
        }

    df = df.drop(columns=['ocsvm'])
    return result


# param_grid_ocsvm = {
#     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
#     'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
#     'nu': [0.01, 0.05, 0.1, 0.2, 0.5, 0.0092, 0.02, 0.03, 0.035, 0.04, 0.06, 0.07, 0.08]
# }
#
# param_combinations_ocsvm = [(kernel, gamma, nu)
#                             for kernel in param_grid_ocsvm['kernel']
#                             for gamma in param_grid_ocsvm['gamma']
#                             for nu in param_grid_ocsvm['nu']]


param_grid_ocsvm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'coef0': [0.0, 0.1, 0.5, 1.0],
    'tol': [1e-3, 1e-4, 1e-5],
    'nu': [0.01, 0.05, 0.1, 0.2, 0.5],
    'shrinking': [True, False],
    'cache_size': [200, 500],
    'max_iter': [-1, 100, 500, 1000]
}

param_combinations_ocsvm = [(kernel, degree, gamma, coef0, tol, nu, shrinking, cache_size, max_iter)
                            for kernel in param_grid_ocsvm['kernel']
                            for degree in param_grid_ocsvm['degree']
                            for gamma in param_grid_ocsvm['gamma']
                            for coef0 in param_grid_ocsvm['coef0']
                            for tol in param_grid_ocsvm['tol']
                            for nu in param_grid_ocsvm['nu']
                            for shrinking in param_grid_ocsvm['shrinking']
                            for cache_size in param_grid_ocsvm['cache_size']
                            for max_iter in param_grid_ocsvm['max_iter']]


if __name__ == '__main__':

    df = pd.read_csv(filepath_or_buffer="../PicoDataSetKerberos.csv")

    # avem o diferenta intre service 1206 si client 1097 : KDC_ERR_BADOPTION 91 + KDC_ERR_S_PRINCIPAL_UNKNOWN 12 +KRB_AP_ERR_TKT_EXPIRED 2+4 nan ex linia 2056
    df.dropna(subset=['client'], inplace=True)
    df.info()
    df.reset_index(drop=True, inplace=True)
    df.info()

    # df["uid"] #este unic pentru fiecare log

    # Excutand functia count_apparances pe coloana "id.resp_h" ne returneaza lista cu un singur element si anume [10.99.99.5] reprezentand adresa serverului pe care ruleaza protocolul kerberos din asta rezulta ca nu avem mai mult servere de kerberos coloana aceasta nu aduce informatii suplimentare <=>  nu vom folosi coloana
    df = df.drop(columns=['id.resp_h'])

    # Dupa ce stergem liniile dupa existeanta sa nu a service verificam id.resp_p cu functia appearences vedem ca ne returneaza doar lista cu elementrul 88 [88] => ca coloana id.resp_p nu aduce nici o informatie utila <=> nu o vom folosi
    df = df.drop(columns=['id.resp_p'])

    df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
    df['ts'] = df['ts'].astype('int64')  ##pt isforest

    df['id.orig_h'] = df['id.orig_h'].apply(ipv4_address_to_int)
    print_column_info(df, 'id.orig_h')

    df.dropna(subset=['till'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['till'] = pd.to_datetime(df['till']).dt.tz_localize(None)
    df['till'] = df['till'].astype('int64')  ####pt isforest
    print_column_info(df, 'till')

    df.dropna(subset=['success'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['success'] = df['success'].apply(bool_to_int)
    print_column_info(df, 'success')

    df['forwardable'] = df['forwardable'].apply(bool_to_int)
    print_column_info(df, 'forwardable')

    df['renewable'] = df['renewable'].apply(bool_to_int)
    print_column_info(df, 'renewable')

    process_column(df, "request_type")
    process_column(df, "cipher")
    process_column(df, "error_msg")
    process_column(df, "service")
    process_column(df, "client")


    df_anomaly = pd.read_csv(filepath_or_buffer="../Anomaly_Set.csv")
    list_uid_anomaly = list(df_anomaly["uid"])

    features = ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'forwardable', 'renewable', 'error_msg', 'client']

    list_features = [
        ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'forwardable',
         'renewable', 'error_msg', 'client']
        # ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'forwardable',
        #  'renewable', 'error_msg'],
        # ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'error_msg', 'client'],
        # ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'error_msg', 'client'],
        # ['ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'error_msg'],
        # ['id.orig_h', 'service', 'success', 'error_msg', 'client']
    ]

    header_ocsvm = ['ratio', 'uid_if_nou_gasite', 'score', 'total_nr_valori_din_set', 'precision', 'recall', 'fpr', 'f1_score', 'kernel', 'degree', 'gamma', 'coef0', 'tol', 'nu', 'shrinking', 'cache_size', 'max_iter',
                    'ts', 'id.orig_h', 'id.orig_p', 'request_type', 'service', 'success', 'till', 'cipher', 'forwardable', 'renewable', 'error_msg', 'client',
                    'CG2fHGatKXKTpsyh6', 'CRUEAq4YtJkoBSShid', 'CTk8QH20I5oC7wqQN9', 'C5XKvBZyCZ4Bpic65', 'CvE9GVJpVxLGrTsO2', 'CJQGFI1WYwA4hnajS4', 'Cdo3iN17MkD2D4SSKk', 'CaKSyP3tBxtPmUfbs1', 'C1jGntzIYAIF8NB33', 'C9qYNL1PMeaOlhux19']

    start_time = time.time()

    csvfile = open(file="OCSVMResults.csv", mode='w+', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(header_ocsvm)
    csvfile.close()

    manager = Manager()
    lock = manager.Lock()

    results_ocsvm = Parallel(n_jobs=-3)(
        delayed(process_parameters_ocsvm)(kernel, degree, gamma, coef0, tol, nu, shrinking, cache_size, max_iter, df, features_item,features, list_uid_anomaly, lock)
        for (kernel, degree, gamma, coef0, tol, nu, shrinking, cache_size, max_iter) in param_combinations_ocsvm
        for features_item in list_features
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    csvfile.close()


