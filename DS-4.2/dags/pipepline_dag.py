from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import pandas as pd

import SakharnyiMLToolkit
from SakharnyiMLToolkit import Feture_ext
from SakharnyiMLToolkit import Model_val

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
    description='This DAG is designed to run the entire pipeline',
    schedule=None,
)

#work
def process_data(path_to_data_csv_file, path_to_shops_csv, path_to_items_csv, months_to_train, month_to_test,lags_list, **kwargs):
    import pandas as pd
    
    data = pd.read_csv(path_to_data_csv_file, low_memory=False)

    train=data[data.is_train == 1]
    test=data[data.is_train == 0]
    del data

    feture_ext = Feture_ext(test = test, train = train)
    # use the make_data_cool method to process data
    feture_ext.operate_data_pipeline(lags_list, path_to_shops_csv, path_to_items_csv)
    processed_data = feture_ext.get_data()
    
    del feture_ext

    train = processed_data[processed_data.date_block_num <= months_to_train]
    test = processed_data[(processed_data.date_block_num > months_to_train)&(processed_data.date_block_num <= (months_to_train+month_to_test))]
    # Store processed data in xcom
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

#work
process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    op_kwargs={'path_to_data_csv_file': '/home/siarhei/Programming/ML/internship/DS-4.2/data/kaggle_data.csv',
               'path_to_shops_csv': '/home/siarhei/Programming/ML/Data/Predict Future Sales/shops.csv',
               'path_to_items_csv': '/home/siarhei/Programming/ML/Data/Predict Future Sales/items.csv',
               'months_to_train': 30,
               'month_to_test':3,
               'lags_list': [1]},
    dag=dag,
)


#work
def run_saved_model(path_to_model, **kwargs):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor().load_model(path_to_model)

    import json
    processed_data = pd.DataFrame.from_dict(json.loads(kwargs['ti'].xcom_pull(key = 'processed_data', task_ids='process_data'))['test'])
    #add training sequence
    processed_data = processed_data[['date_block_num', 'shop_id', 'item_id', 'item_category_id',
       'item_cnt_month_lag_1', 'shop_age_in_months', 'item_age', 'city_code_y',
       'shop_type_code_y', 'item_name_group_y', 'ID', 'item_cnt_month']]
    
    predictions = model.predict(processed_data.drop(columns=['item_cnt_month','ID']))

    import os
    model_name = os.path.basename(path_to_model).split('.')[0]
    kwargs['ti'].xcom_push(key='predictions', value=predictions.tolist())
    kwargs['ti'].xcom_push(key='model_name', value=model_name)

#work
run_saved_model_task = PythonOperator(
    task_id='run_saved_model',
    python_callable=run_saved_model,
    provide_context=True,
    op_kwargs={'path_to_model':'/home/siarhei/Programming/ML/internship/DS-4.2/data/catboost_model1.bin'},
    dag=dag,
)


#доработать Model_val / заменить его на на error_analysis
def get_metrics(**kwargs):
    import json

    predictions = kwargs['ti'].xcom_pull(key='predictions',task_ids="run_saved_model")
    processed_data = pd.DataFrame.from_dict(json.loads(kwargs['ti'].xcom_pull(key = 'processed_data', task_ids='process_data'))['test'])
    model_name = kwargs['ti'].xcom_pull(key='model_name',task_ids="run_saved_model")
    test_y = processed_data['item_cnt_month']

    error_analysis = Model_val(avoid_model=True, model = None, test_y = test_y,\
                               predicted_y = predictions, test_x = None)
    # write the results of error analysis to files
    error_analysis.get_model_results()
    metrics = error_analysis.get_metrics()
    print(metrics)
    #import json 
    #with open(model_name+'.json', "w") as outfile: #добавить имя модели
    #    json.dump(metrics, outfile)
    

#таска для получения метрик модели + анализа ошибок
get_metrics_task = PythonOperator(
    task_id='get_metrics',
    python_callable=get_metrics,
    provide_context=True,
    op_kwargs={},
    dag=dag,
)

#work && dvc pull
pull_data_task = BashOperator(
    task_id='pull_data',
    bash_command='cd /home/siarhei/Programming/ML/internship/DS-4.2 && pwd',
    dag=dag,
)

#push_results_task = BashOperator(
#    task_id='push_results',
#    bash_command='dvc push',
#    dag=dag,
#)

pull_data_task >> process_data_task >> run_saved_model_task >> get_metrics_task #>> push_results_task
