import os, csv, warnings, time
import pandas as pd
from multiprocessing import Manager
from prophet import Prophet
from joblib import Parallel, delayed
from collections import defaultdict
import hashlib
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", category=FutureWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def print_process_anomalies(id,list_uid_prop_conex_number, list_uid_anomaly, rowname, key,type_pross,in_wd, u_samp, cps, sps,  sm, cr, g, nc, minute,
                      lock):
    uid_identified_as_anomalies = [item for item in list_uid_prop_conex_number if item in list_uid_anomaly]

    score = len(uid_identified_as_anomalies)

    if score > 0:

        total_lenght_dataset = len(df)
        true_positive = len(uid_identified_as_anomalies)
        false_positive = len(list_uid_prop_conex_number) - true_positive
        false_negative = len(list_uid_anomaly) - true_positive
        true_negative = total_lenght_dataset - len(list_uid_anomaly) - false_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        f_score = 2 * ((precision * recall) / (precision + recall))

        print(f"{precision} precision")
        print(f"{recall} recall")
        print(f"{fpr} recall")
        print(f"{f_score} f1_score")
        print(f"{len(uid_identified_as_anomalies)} total anomalies find")
        print(f"{score} total corect anomalies")


        list_row = [
            id,
            len(list_uid_prop_conex_number) / score,
            len(uid_identified_as_anomalies),
            score,
            len(list_uid_prop_conex_number),
            str(rowname),
            str(key),
            precision,
            recall,
            fpr,
            f_score,
            type_pross,
            in_wd,
            u_samp,
            cps,
            sps,
            sm,
            cr,
            g,
            nc,
            minute
        ]


        list01_an = [1 if item in uid_identified_as_anomalies else 0 for item in list_uid_anomaly]
        list_row.extend(list01_an)

        with lock:
            with open(file="PropResults.csv", mode='a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list_row)
                csvfile.close()

def custom_transform_dict(data, rowname):

    result = []
    key_list = []

    for (index, rows) in data:

        c_dict=defaultdict(int)

        for row in rows[rowname]:
            if str(row) == 'nan':
                key_list.append(str(row))
            else:
                key_list.append(row)

            c_dict[str(row)] +=1

        for key, value in c_dict.items():
            result.append((index, rowname, str(key), value))

    return pd.DataFrame(result, columns=['ds', 'rowname','atribute', 'y']), list(set(key_list))


def generate_numeric_id(in_wd, u_samp, changepoint_prior_scale, seasonality_prior_scale,
                         seasonality_mode, changepoint_range,
                        growth, n_changepoints,rowname, times):

    param_str = f"{rowname}_{times}_iw{in_wd}_us{u_samp}_cp{changepoint_prior_scale}_sp{seasonality_prior_scale}_{seasonality_mode}_cr{changepoint_range}_{growth}_nc{n_changepoints}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    numeric_id = int(param_hash, 16) % (10 ** 10)

    return numeric_id


def process_parameters(in_wd, u_samp,cps,sps,sm,cr,g,nc,minute,rowname,df,df_anomaly,lock):

    uid_eveniment=[]
    uid_interval=[]

    df['ds'] = df['ts'].dt.floor(minute)
    df_anomaly['ds'] = df_anomaly['ts'].dt.floor(minute)

    list_ds_anomaly = list(df_anomaly["ds"])
    list_uid_anomaly = list(df_anomaly["uid"])

    group = df.groupby('ds')
    result, key_list = custom_transform_dict(group, rowname=rowname)

    id = generate_numeric_id(in_wd, u_samp, cps, sps,  sm, cr, g, nc, rowname, minute)

    for key in key_list:

        result_atribute = result[result['atribute'] == str(key)]


        if g == 'logistic':
            max_capacity = result_atribute['y'].max() * 1.3
            result_atribute['cap']=max_capacity


        if len(result_atribute)<2:

            list_uid_prop_conex_number = df['uid'][df['ds'].isin(result_atribute['ds'].tolist())][df[rowname] == key].tolist()# specific event
            for element in list_uid_prop_conex_number:
                if element not in uid_eveniment:
                    uid_eveniment.append(element)
            print_process_anomalies(id, list_uid_prop_conex_number, list_uid_anomaly, rowname, key, "eveniment_nr",
                                    in_wd, u_samp, cps, sps, sm, cr, g, nc, minute, lock)


            list_uid_prop_conex_number = df['uid'][df['ds'].isin(result_atribute['ds'].tolist())].tolist() #specific time frame
            for element in list_uid_prop_conex_number:
                if element not in uid_interval:
                    uid_interval.append(element)
            print_process_anomalies(id, list_uid_prop_conex_number, list_uid_anomaly, rowname, key, "interval_nr",
                                    in_wd, u_samp, cps, sps, sm, cr, g, nc, minute, lock)

            continue


        model_prophet = Prophet(
            interval_width=in_wd,
            uncertainty_samples=u_samp,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            #holidays_prior_scale=hps,#nu sunt sarbatori in perioada din data set nici in sun nici in romania
            seasonality_mode=sm,
            changepoint_range=cr,
            growth=g,
            n_changepoints=nc
        )

        model_prophet.fit(result_atribute)

        forecast = model_prophet.predict(result_atribute)
        for row in forecast.itertuples(index=False):
            print(row)


        performance = pd.merge(result_atribute[['ds', 'y']], forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
        performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper)) else 0, axis=1)


        print(f"numarul de posibile anomalii {performance['y'][performance['anomaly']==1].sum()}")
        performance['correct_anomaly'] = performance['ds'][performance['anomaly'] == 1].isin(list_ds_anomaly)

        print(performance['correct_anomaly'].value_counts())

        anomalies_all = performance['ds'][performance['anomaly'] == 1].tolist()


        list_uid_prop_conex_number=df['uid'][df['ds'].isin(anomalies_all)].tolist() #pentru interval de timp
        for element in list_uid_prop_conex_number:
            if element not in uid_interval:
                uid_interval.append(element)
        print_process_anomalies(id,list_uid_prop_conex_number, list_uid_anomaly, rowname, key,"interval",in_wd, u_samp, cps, sps, sm, cr, g, nc, minute,lock)


        list_uid_prop_conex_number = df['uid'][df['ds'].isin(anomalies_all)][df[rowname]==key].tolist()#pentru evenimentele de timp din acea perioasa
        for element in list_uid_prop_conex_number:
            if element not in uid_eveniment:
                uid_eveniment.append(element)
        print_process_anomalies(id,list_uid_prop_conex_number, list_uid_anomaly, rowname, key,"eveniment",in_wd, u_samp, cps, sps,  sm, cr, g, nc, minute,lock)

        # Plot the forecast
        # fig = model_prophet.plot(performance)
        # plt.title("Label "+str(rowname)+" "+str(key), pad=20)
        # plt.scatter(performance[performance['correct_anomaly'] == False]['ds'], performance[performance['correct_anomaly'] == False]['y'], color='blue',
        #             label='False Positive')
        #
        # plt.scatter(performance[performance['correct_anomaly'] == True]['ds'], performance[performance['correct_anomaly'] == True]['y'], color='green',
        #             label='True Positive')
        # plt.tight_layout(rect=[0, 0, 1, 0.93])
        # plt.legend()
        # plt.show()


    print_process_anomalies(id, uid_eveniment, list_uid_anomaly, rowname, "all", "eveniment_global",in_wd, u_samp, cps, sps,
                                sm, cr, g, nc, minute, lock)

    print_process_anomalies(id, uid_interval, list_uid_anomaly, rowname, "all", "interval_global",in_wd, u_samp, cps, sps,
                             sm, cr, g, nc, minute, lock)



if __name__ == '__main__':

    df = pd.read_csv(filepath_or_buffer="PicoDataSetKerberos.csv")

    df.dropna(subset=['client'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.dropna(subset=['till'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.dropna(subset=['success'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.info()

    df["ts"]=pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.sort_values(by='ts')

    df_anomaly = pd.read_csv(filepath_or_buffer="Anomaly_Set.csv")
    df_anomaly["ts"] = pd.to_datetime(df_anomaly["ts"]).dt.tz_localize(None)
    df_anomaly = df_anomaly.sort_values(by='ts')

    df_anomaly["ds"] = pd.to_datetime(df_anomaly["ts"]).dt.tz_localize(None)

    header = ['id','ratio', 'uid_if_nou_gasite', 'score', 'total_nr_valori_din_set','status','value','precision','recall','fpr','f1_score',
              'type_pross',
              'interval_width', 'uncertainty_samples','changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode','changepoint_range','growth','n_changepoints', 'minutes',
              'CG2fHGatKXKTpsyh6', 'CRUEAq4YtJkoBSShid', 'CTk8QH20I5oC7wqQN9', 'C5XKvBZyCZ4Bpic65', 'CvE9GVJpVxLGrTsO2',
              'CJQGFI1WYwA4hnajS4', 'Cdo3iN17MkD2D4SSKk', 'CaKSyP3tBxtPmUfbs1', 'C1jGntzIYAIF8NB33',
              'C9qYNL1PMeaOlhux19']

    #
    # param_grid = {
    #
    #     'rowname':['id.resp_h','success','client'],
    #     'times':["1min","2min","5min","10min","30min","1h","2h","12h","1d"],
    #     'interval_width': [0.95, 0.90, 1],
    #     'uncertainty_samples': [1000, 10],
    #     'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    #     'seasonality_prior_scale':[0.1, 1.0, 10.0],
    #     #'holidays_prior_scale': [0.1, 1.0, 10.0], nu e vacanta in intervalul ala nici sua nici romania
    #     'seasonality_mode': ['additive', 'multiplicative'],
    #     'changepoint_range': [0.8, 0.9, 0.95],
    #     'growth': ['linear', 'logistic'],
    #     'n_changepoints': [10, 25, 50]
    # }

    #best parameters
    param_grid = {
        'rowname': ['id.resp_h','client'],
        'times': ["1min","2min","5min","10min"],
        'uncertainty_samples': [ 10],
        'interval_width': [0.95],
        'changepoint_prior_scale': [ 0.01, 0.5],
        'seasonality_prior_scale': [0.1],
        'seasonality_mode': ['additive'],
        'changepoint_range': [0.8,0.9],
        'growth': ['logistic'],
        'n_changepoints': [50]
    }


    param_combinations = [(rowname,times,in_wd, u_samp,cps,sps,sm,cr,g,nc)

                          for rowname in param_grid['rowname']
                          for times in param_grid['times']
                          for in_wd in param_grid['interval_width']
                          for u_samp in param_grid['uncertainty_samples']
                          for cps in param_grid['changepoint_prior_scale']
                          for sps in param_grid['seasonality_prior_scale']
                          for sm in param_grid['seasonality_mode']
                          for cr in param_grid['changepoint_range']
                          for g in param_grid['growth']
                          for nc in param_grid['n_changepoints']]

    start_time = time.time()

    csvfile = open(file="PropResults.csv", mode='w+', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(header)
    csvfile.close()

    manager = Manager()
    lock = manager.Lock()

    results = Parallel(n_jobs=-3)(
        delayed(process_parameters)(in_wd, u_samp,cps, sps,  sm, cr, g, nc, times,rowname, df,df_anomaly,lock)
        for (rowname,times,in_wd, u_samp,cps,sps,sm,cr,g,nc) in param_combinations
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time: {total_time:.6f} seconds")
    csvfile.close()