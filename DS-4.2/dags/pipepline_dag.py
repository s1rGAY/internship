from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import pandas as pd

import SakharnyiMLToolkit
from SakharnyiMLToolkit import Feture_ext
from SakharnyiMLToolkit import Model_val

#configuration
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

#path
path_to_workspace = config.get('path_env', 'path_to_workspace')
path_to_dvc = config.get('path_env', 'path_to_dvc')
path_to_data_csv_file = config.get('path_env', 'path_to_data_csv_file')
path_to_shops_csv = config.get('path_env', 'path_to_shops_csv')
path_to_items_csv = config.get('path_env', 'path_to_items_csv')
path_to_model = config.get('path_env', 'path_to_model')

#model_env
months_to_train = config.getint('model_env', 'months_to_train')
month_to_test = config.getint('model_env', 'month_to_test')
lags_list = config.getint('model_env', 'lags_list')

#neptune_env
project = config.get('neptune_env', 'project')
api_token = config.get('neptune_env', 'api_token')


default_args = {
    'owner': 'you',
    'start_date': datetime(2023, 2, 12),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'pipeline_dag',
    default_args=default_args,
    description='This DAGestanec is designed to run the entire pipeline',
    schedule=None,
)


def process_data(path_to_data_csv_file, path_to_shops_csv, path_to_items_csv, months_to_train, month_to_test,lags_list, **kwargs):
    import pandas as pd
    
    data = pd.read_csv(path_to_data_csv_file, low_memory=False)

    train=data[data.is_train == 1]
    test=data[data.is_train == 0]
    del data

    feture_ext = Feture_ext(test = test, train = train)

    feture_ext.operate_data_pipeline(lags_list, path_to_shops_csv, path_to_items_csv)
    processed_data = feture_ext.get_data()
    
    del feture_ext

    train = processed_data[processed_data.date_block_num <= months_to_train]
    test = processed_data[(processed_data.date_block_num > months_to_train)&(processed_data.date_block_num <= (months_to_train+month_to_test))]

    data = dict()
    print(train.columns)
    data['train'] = train.to_dict(orient='records')
    data['test'] = test.to_dict(orient='records')

    del train
    del test

    import json 
    data = json.dumps(data)
    print(data)
    kwargs['ti'].xcom_push(key='processed_data', value=data)



process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    op_kwargs={'path_to_data_csv_file': path_to_data_csv_file,
               'path_to_shops_csv': path_to_shops_csv,
               'path_to_items_csv': path_to_items_csv,
               'months_to_train': months_to_train,
               'month_to_test':month_to_test,
               'lags_list': [lags_list]},
    dag=dag,
)



def run_saved_model(path_to_model, **kwargs):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor().load_model(path_to_model)

    import json
    processed_data = pd.DataFrame.from_dict(json.loads(kwargs['ti'].xcom_pull(key = 'processed_data', task_ids='process_data'))['test'])

    processed_data = processed_data[['date_block_num', 'shop_id', 'item_id', 'item_category_id',
       'item_cnt_month_lag_1', 'shop_age_in_months', 'item_age', 'city_code_y',
       'shop_type_code_y', 'item_name_group_y', 'ID', 'item_cnt_month']]
    
    predictions = model.predict(processed_data.drop(columns=['item_cnt_month','ID']))

    import os
    model_name = os.path.basename(path_to_model).split('.')[0]
    kwargs['ti'].xcom_push(key='predictions', value=predictions.tolist())
    kwargs['ti'].xcom_push(key='model_name', value=model_name)


run_saved_model_task = PythonOperator(
    task_id='run_saved_model',
    python_callable=run_saved_model,
    provide_context=True,
    op_kwargs={'path_to_model':path_to_model},
    dag=dag,
)


pull_data_task = BashOperator(
    task_id='pull_data',
    bash_command='cd {} && pwd && dvc pull'.format(path_to_workspace),
    dag=dag,
)


def error_analysis_and_neptune(**kwargs):
    import json
    import matplotlib.pyplot as plt
    import neptune.new as neptune
    from neptune.new.types import File
    import numpy as np
    import pandas as pd
    import seaborn as sns

    run = neptune.init(
    project=project,
    api_token=api_token,
    )
    
    def plot_and_log(real_values, predicted_values, image_name='plot_and_log'):    
        plt.scatter(real_values, predicted_values)

        plt.xlabel("Real Values")
        plt.ylabel("Predicted Values")
        plt.title("Real vs. Predicted Values")
        
        image = File.as_image(plt.gcf())
        run[f'{image_name}'].log(image)
    
    def plot_predicted_vs_real(predicted_values, real_values, image_name='predicted_vs_real'):
        
        bins = np.arange(0, 300, 1)#300->2
        bin_centers = (bins[1:] + bins[:-1]) / 2
        predicted_means, _, _ = plt.hist(real_values, bins=bins, weights=predicted_values, density=True)
        
        predicted_stdevs, _, _ = plt.hist(real_values, bins=bins, weights=predicted_values**2, density=True)
        predicted_stdevs = np.sqrt(predicted_stdevs - predicted_means**2)
        
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 25))
            
        for i, ax in enumerate(axes.flat):
            start_idx = i * 10
            end_idx = (i + 1) * 10
            x_vals = bin_centers[start_idx:end_idx]
            y_vals = predicted_means[start_idx:end_idx]
            y_stdevs = predicted_stdevs[start_idx:end_idx]
            ax.errorbar(x_vals, y_vals, yerr=y_stdevs, fmt='o', color='red', capsize=5, capthick=2)
            ax.set_xlabel('Real Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Subplot {i+1}')
        
        image = plt.gcf()
        run[image_name].log(neptune.types.File.as_image(image))
    
    
    def plot_normalized_error_neptune(real_values, predicted_values, start, stop, step, image_name='normalized_error_neptune'):
        
        bins = np.arange(start, stop, step)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        predicted_means, _, _ = plt.hist(real_values, bins=bins, weights=predicted_values, density=True)
        predicted_stdevs, _, _ = plt.hist(real_values, bins=bins, weights=predicted_values**2, density=True)
        predicted_stdevs = np.sqrt(predicted_stdevs - predicted_means**2)
    
    
        real_values_norm = (real_values - np.mean(real_values)) / np.std(real_values)
        predicted_values_norm = (predicted_values - np.mean(predicted_values)) / np.std(predicted_values)
    

        predicted_stdevs = np.repeat(predicted_stdevs, 2)
    

        predicted_stdevs_norm = np.std(predicted_values_norm)
    

        real_stdevs_norm = np.std(real_values_norm)
    

        plt.errorbar(real_values_norm, predicted_values_norm, yerr=predicted_stdevs_norm, fmt='o', color='red', capsize=5, capthick=2)
    
  
        plt.xlabel('Real Values (Normalized)')
        plt.ylabel('Predicted Values (Normalized)')
                
    
        image = plt.gcf()
        run[image_name].log(neptune.types.File.as_image(image))
        

    def plot_mean_predicted_values(predicted_values, real_values, image_name='mean_predicted_values'):

        df = pd.DataFrame({'real': real_values, 'predicted': predicted_values})
        grouped = df.groupby('real').mean()
        real_mean_predicted = grouped['predicted'].values
        real_bins = grouped.index.values
        num_subplots = int(np.ceil(len(real_bins) / 25))


        sns.set(style='ticks', font_scale=1.2, palette='Dark2')
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(12, 8*num_subplots))
        #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        #fig, axes = plt.subplots(3, 2)

        # Plot a subset of the data on each subplot
        for i in range(num_subplots):
            start_idx = i * 25
            end_idx = min((i + 1) * 25, len(real_bins))
            ax = axes[i]
            ax = axes
            sns.lineplot(x=real_bins[start_idx:end_idx], y=real_mean_predicted[start_idx:end_idx], ax=ax)
            ax.set_xlabel('Real Values')
            ax.set_ylabel('Mean Predicted Values')
            ax.set_title(f'Subplot {i+1}')
            ax.grid()
        
        image = plt.gcf()
        run[image_name].log(neptune.types.File.as_image(image))

        
    def metrics(predicted_values, real_values):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(real_values, predicted_values)
        rmse = pow(mse,0.5)
        mae = mean_absolute_error(real_values, predicted_values)
        r2_scoree = r2_score(real_values, predicted_values)

        run['mse'] = mse
        run['rmse'] = rmse
        run['mae'] = mae
        run['r2_score'] = r2_scoree
    
    predicted_values = np.array(kwargs['ti'].xcom_pull(key='predictions',task_ids="run_saved_model"))
    real_values = pd.DataFrame.from_dict(json.loads(kwargs['ti'].xcom_pull(key = 'processed_data', task_ids='process_data'))['test'])
    real_values = real_values['item_cnt_month']
    
    #real_values = kwargs['ti'].xcom_pull(key = 'predictions', task_ids='run_saved_model')
    #check kaggle settings
    print(f'Working with : plot_and_log')
    plot_and_log(predicted_values=predicted_values, real_values=real_values)
    print(f'Working with : plot_predicted_vs_real')
    plot_predicted_vs_real(predicted_values, real_values)
    print(f'Working with : plot_normalized_error_neptune')
    plot_normalized_error_neptune(real_values, predicted_values, 0, 300, 1)
    print(f'Working with : plot_mean_predicted_values')
    plot_mean_predicted_values(predicted_values, real_values)
    print(f'Working with : metrics')
    metrics(predicted_values, real_values)
    run.stop()


error_analysis_and_neptune_task = PythonOperator(
    task_id='error_analysis_and_neptune',
    python_callable=error_analysis_and_neptune,
    provide_context=True,
    op_kwargs={},
    dag=dag,
)

pull_data_task >> process_data_task >> run_saved_model_task >> error_analysis_and_neptune_task
