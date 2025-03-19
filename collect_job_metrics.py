import pandas as pd
import conf_2
import os


def collect_job_metrics():
    print("collecting job metrics")
    path = conf.output_metrics_dir
    df = pd.DataFrame()

    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    for file in csv_files:
        print(file)
        df_temp = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df,df_temp],  ignore_index=True)

    df = df.sort_values('Metric (MSE)', ascending=True)

    df.to_csv(os.path.join(path, 'job_train_metrics.csv'), index=False, mode='a', header=False)
