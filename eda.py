import pandas as pd
import numpy as np
import mysql.connector
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv(r"C:\Users\şerefcanmemiş\Documents\projects_2\data_handle\.env")

user = os.getenv('user')
password = os.getenv('password')
host = os.getenv('host')
database = os.getenv('database')
folder = os.getenv('folder')

engine = create_engine(
    f'mysql+pymysql://{user}:{password}@{host}:3306/{database}?charset=utf8mb4',
    connect_args={"local_infile": 1}
)

tables = [
    'application_combined',
    'bureau',
    'bureau_balance',
    'credit_card_balance',
    'installments_payments',
    'pos_cash_balance',
    'previous_application',
    'sample_submission'
]
dfs = {}

for table in tables:
    dfs[table] = pd.read_sql(f"select * from {table}",con = engine)
    print(f'{table} successfully added into the dfs')
    
print(dfs.keys())
dfs['application_train'] = dfs['application_combined'][dfs['application_combined']['TARGET'].notna()].copy()
dfs['application_test'] = dfs['application_combined'][dfs['application_combined']['TARGET'].isna()].copy()

dfs.pop('application_combined')
print(dfs.keys())

# Checking null values
tables = [
    'application_train',
    'application_test',
    'bureau',
    'bureau_balance',
    'credit_card_balance',
    'installments_payments',
    'pos_cash_balance',
    'previous_application',
    'sample_submission'
]
def null_percentage():
    for table in tables:
        namesofnull = dfs[table].isna().sum()[dfs[table].isna().sum() > 0].sort_values(ascending = False).index
        numberofnull = dfs[table].isna().sum()[dfs[table].isna().sum() > 0].sort_values(ascending = False).values
        percentage = round(100 * dfs[table].isna().sum()[dfs[table].isna().sum() > 0] / dfs[table].shape[0],2).sort_values(ascending = False)
        df_nulls = pd.DataFrame()
        df_nulls = pd.DataFrame({
        table: namesofnull,
        'null_count': numberofnull,
        'null_percentage': percentage
    })
        print(df_nulls[df_nulls['null_percentage'] > 0])
    
# We have null values only in application_train and application_test, others don't have NA values

def drop_null_colums(percentage):
    for table in tables:
        null_percentage = round(100 * dfs[table].isna().sum()[dfs[table].isna().sum() > 0] / dfs[table].shape[0],2).sort_values(ascending = False)
        col_to_drop = null_percentage[null_percentage > percentage].index.tolist()
        
        if 'EXT_SOURCE_1' in col_to_drop:
            col_to_drop.remove('EXT_SOURCE_1')
        
        if col_to_drop:
            dfs[table] = dfs[table].drop(col_to_drop, axis = 1)
            print(f'*{table}* successfully dropped {col_to_drop}')
        else:
            print(f'*{table}* no column to drop')
            
null_percentage()

app_train = dfs['application_train'].copy()
app_test = dfs['application_test'].copy()


app_train['TARGET'].value_counts()

app_train.dtypes.value_counts()

app_train.select_dtypes('object').apply(pd.Series.nunique)

# if object nunique > 2 onehotencoder else labelencoder