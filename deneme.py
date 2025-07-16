import pandas as pd
import mysql.connector
import os
from sqlalchemy import create_engine

user = os.getenv('user')
password = os.getenv('password')
host = os.getenv('host')
database = os.getenv('database')
folder = os.getenv('folder')


engine = create_engine(
    f'mysql+pymysql://{user}:{password}@{host}:3306/{database}?charset=utf8mb4',
    connect_args={"local_infile": 1}
)

folder = r"C:/Users/şerefcanmemiş/Downloads/home-credit-default-risk"
files_to_tables = {
    "previous_application.csv":"previous_application",
    "sample_submission.csv":"sample_submission",
    "installments_payments.csv":"installments_payments",
    "credit_card_balance.csv":"credit_card_balance",
    "bureau_balance.csv":"bureau_balance",
    "bureau.csv":"bureau",
    "application_train.csv":"application_train",
    "application_test.csv":"application_test",
    "POS_CASH_balance.csv":"pos_cash_balance"
}
dataframes = {}
for csv, table_name in files_to_tables.items():
    path = os.path.join(folder, csv).replace('\\','/')
    dataframes[table_name] = pd.read_csv(path)

application_train = dataframes['application_train'].copy()
application_test = dataframes['application_test'].copy()
application_combined = pd.concat([application_train,application_test])
application_combined.to_sql('application_combined', con=engine, index=False, if_exists='replace')

set(dataframes['application_train']) - set(dataframes['application_test']) # TARGET

set(dataframes['bureau']) & set(dataframes['application_train'])
set(dataframes['bureau']) & set(dataframes['bureau_balance'])
print('SK_ID_CURR is a primary key for *application_train* and foreign key for *bureau*')
print('SK_ID_BUREAU is a primary key for *bureau* and foreign key for *bureau_balance*')

