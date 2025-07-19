import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# load .env
load_dotenv(r"C:\Users\şerefcanmemiş\Documents\projects_2\data_handle\.env")

user = os.getenv('user')
password = os.getenv('password')
host = os.getenv('host')
database = os.getenv('database')
folder = os.getenv('folder')

# start the engine
engine = create_engine(
    f'mysql+pymysql://{user}:{password}@{host}:3306/{database}?charset=utf8mb4',
    connect_args={"local_infile": 1}
)

# define tables
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

# Extract tables from mysql to python dictionaries
for table in tables:
    dfs[table] = pd.read_sql(f"select * from {table}",con = engine)
    print(f'{table} successfully added into the dfs')
    
print(dfs.keys())

# divide into train and test
dfs['application_train'] = dfs['application_combined'][dfs['application_combined']['TARGET'].notna()].copy()
dfs['application_test'] = dfs['application_combined'][dfs['application_combined']['TARGET'].isna()].copy()
dfs.pop('application_combined')


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

# Check the null percentage of columns
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
null_percentage()
    
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
            

# Copying our train and test tables
app_train = dfs['application_train'].copy()
app_test = dfs['application_test'].copy()
app_train = app_train[app_test.columns]

# Check if the 2 tables columns are the same
if list(app_train.columns) == list(app_test.columns):
    print('test and train columns are equal')
else:
    print('test and train columns are not equal!')
    
# Check if the values in column are same in both of the tables (bc they are going to be problem in onehotencoding)
app_test['CODE_GENDER'].value_counts()
app_train['CODE_GENDER'].value_counts() # just 4 XNA lets drop them
app_train = app_train[app_train['CODE_GENDER'] != 'XNA']

app_train['NAME_FAMILY_STATUS'].value_counts() # just 2 unkown lets also drop them
app_test['NAME_FAMILY_STATUS'].value_counts()
app_train = app_train[app_train['NAME_FAMILY_STATUS'] != 'Unknown']

app_train['NAME_INCOME_TYPE'].value_counts() # 5 maternity leave
app_test['NAME_INCOME_TYPE'].value_counts()


# Check class imbalance
app_train['TARGET'].value_counts()

# Check data types counts
app_train.dtypes.value_counts()

# Check unique count in categorical data
app_train.select_dtypes('object').apply(lambda col: col.nunique()).sort_values(ascending = False)

# if object nunique > 2 onehotencoder else labelencoder
print(app_train.select_dtypes('object'))

def labelencode_df():
    le = LabelEncoder()
    for col in app_train.select_dtypes('object'):
        if app_train[col].nunique() <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            
            if set(app_test[col].dropna().unique()).issubset(le.classes_):
                app_test[col] = le.transform(app_test[col])
            else:
                print(f'{col} has unseen labels in test skipping encoding')


def onehotencode_df(df):
    encoded_df = pd.get_dummies(df, dtype='int')
    return encoded_df

labelencode_df()
app_train = onehotencode_df(df = app_train)
app_test = onehotencode_df(df = app_test)

print(app_train.shape) # has 1 extra column we need to figure out how to equalize columns
print(app_test.shape)
set(app_train) - set(app_test) # 'NAME_INCOME_TYPE_Maternity leave'
set(app_test) - set(app_train)

idx_NITM = app_train.columns.get_loc('NAME_INCOME_TYPE_Maternity leave')

if 'NAME_INCOME_TYPE_Maternity leave' not in app_test.columns:
    app_test.insert(idx_NITM, 'NAME_INCOME_TYPE_Maternity leave', pd.Series([0 * len(app_test)]))

set(app_train) - set(app_test)
set(app_test) - set(app_train)

# Columns inconsistency problem solved

corr = app_train.corr()['TARGET'].sort_values(ascending=True)
corr.tail(30)
corr.head(30)

app_train['DAYS_BIRTH_YEARS'] = -app_train['DAYS_BIRTH'] / 365












