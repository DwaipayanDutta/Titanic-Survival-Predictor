from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib

# Airflow arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'titanic_survival_predictor',
    default_args=default_args,
    description='DAG for Titanic Survival Prediction Workflow',
    schedule_interval='@daily',  
)
# Define Python functions for tasks
def load_model(**kwargs):
    model_path = '/model/titanic_model.pkl'  
    model = joblib.load(model_path)
    kwargs['ti'].xcom_push(key='model', value=model)
    print("Model loaded successfully.")

def preprocess_data(**kwargs):
    dataset_url = "https://raw.githubusercontent.com/DwaipayanDutta/Titanic_App/refs/heads/main/Data/titanic.csv"
    data = pd.read_csv(dataset_url)
    features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    features['Sex'] = features['Sex'].map({'male': 0, 'female': 1})
    features['Embarked'] = features['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    kwargs['ti'].xcom_push(key='features', value=features)
    print("Data preprocessing completed.")

def make_prediction(**kwargs):
    ti = kwargs['ti']
    model = ti.xcom_pull(key='model', task_ids='load_model')
    features = ti.xcom_pull(key='features', task_ids='preprocess_data')
    passenger_id = 1 
    passenger_data = features.loc[passenger_id]
    prediction = model.predict([passenger_data])
    print(f"Prediction for Passenger {passenger_id}: {'Survived' if prediction[0] == 1 else 'Not Survived'}")

# Define tasks in the DAG

load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model,
    provide_context=True,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    provide_context=True,
    dag=dag,
)

# Task dependencies
load_model_task >> preprocess_data_task >> make_prediction_task