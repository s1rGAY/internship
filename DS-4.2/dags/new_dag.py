from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_data(**kwargs):
    # Generate some data to push
    data = {'key': 'value'}
    
    # Use xcom_push to send data to the next task
    kwargs['ti'].xcom_push(key='data', value=data)

def pull_data(**kwargs):
    # Use xcom_pull to retrieve data from the previous task
    data = kwargs['ti'].xcom_pull(key='data', task_ids='push_data')
    
    # Print the data to check that it was passed correctly
    print(data)

# Create a DAG object
dag = DAG(
    'new_dag',
    schedule_interval='@once',
    start_date=datetime(2022, 1, 1),
    catchup=False
)

# Create two PythonOperator objects, one to push data and one to pull data
push_data_task = PythonOperator(
    task_id='push_data',
    python_callable=push_data,
    provide_context=True,
    dag=dag
)

pull_data_task = PythonOperator(
    task_id='pull_data',
    python_callable=pull_data,
    provide_context=True,
    dag=dag
)

# Set the push_data_task to run first, and then run the pull_data_task
push_data_task >> pull_data_task
