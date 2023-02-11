from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import pandas as pd

import SakharnyiMLToolkit
from SakharnyiMLToolkit import Feture_ext
from SakharnyiMLToolkit import Model_opt
from SakharnyiMLToolkit import Model_val

default_args = {
    'owner': 'you',
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_dag',
    default_args=default_args,
    description='My sample DAG',
    schedule_interval=timedelta(hours=1),
)


#читаем данные для модификации
def read_csv_file(**kwargs):
    ti = kwargs['ti']
    path_to_file_csv = ti.xcom_pull(task_ids='push_to_xcom')
    df = pd.read_csv(path_to_file_csv)
    return df

#Добвить чтение для модели(по имени)
def run_saved_model(**kwargs):
    # load the model from the file
    model = print('читаем catboost')# load your model
    # get model predictions on processed data
    predictions = model.predict(kwargs['processed_data'])
    return predictions

#доработать Model_val / заменить его на на error_analysis
def get_metrics(**kwargs):
    # create an instance of the error_analysis class
    error_analysis = Model_val()
    # write the results of error analysis to files
    error_analysis.write_data(kwargs['predictions'])

#делаем данные ахуевшими (ДОБАВИТЬ ФИКС ДЛЯ Feture_ext)
def process_data(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='read_csv_file')
    # create an instance of the Feture_ext class
    feture_ext = Feture_ext()
    # use the make_data_cool method to process data
    processed_data = feture_ext.make_data_cool(df)
    ti.xcom_push(key='processed_data', value=processed_data)
    return processed_data

#таска для получения данных их моего CSV (передается дальше в обработку)
read_csv_file_task = PythonOperator(
    task_id='read_csv_file',
    python_callable=read_csv_file,
    op_kwargs={'path_to_csv_file': 'path/to/your/csv/file.csv'},
    dag=dag,
)

#таска для обработки данных (достаю фичи)
process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    op_kwargs={'path_to_csv_file': 'path/to/your/csv/file.csv'},
    dag=dag,
)

#таска для чтения модели и получения предиктов
run_saved_model_task = PythonOperator(
    task_id='run_saved_model',
    python_callable=lambda data, **kwargs: run_saved_model(model_file='my_saved_model', data=data),
    provide_context=True,
    op_kwargs={'data': '{{ ti.xcom_pull(task_ids="run_python_code") }}'},
    dag=dag,
)


#таска для получения метрик модели + анализа ошибок
get_metrics_task = PythonOperator(
    task_id='get_metrics',
    python_callable=get_metrics,
    op_kwargs={'predictions': '{{ ti.xcom_pull(task_ids="run_saved_model") }}'},
    dag=dag,
)


#получаю данные
pull_data_task = BashOperator(
    task_id='pull_data',
    bash_command='dvc pull',
    dag=dag,
)

#пушу данные на облако
push_results_task = BashOperator(
    task_id='push_results',
    bash_command='dvc add ...files... && dvc push',
    dag=dag,
)

pull_data_task >> \
    read_csv_file_task >> \
        process_data_task >> \
            run_saved_model_task >> \
                get_metrics_task >> \
                    push_results_task
