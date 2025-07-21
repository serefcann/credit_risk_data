import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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
def load_tables_from_db():
    for table in tables:
        dfs[table] = pd.read_sql(f"select * from {table}",con = engine)
        print(f'{table} successfully added into the dfs')
        
        # divide into train and test
    dfs['application_train'] = dfs['application_combined'][dfs['application_combined']['TARGET'].notna()].copy()
    dfs['application_test'] = dfs['application_combined'][dfs['application_combined']['TARGET'].isna()].copy()
    dfs.pop('application_combined')

    app_train = dfs['application_train'].copy()
    app_test = dfs['application_test'].copy()
    app_train = app_train[app_test.columns]
    print(dfs.keys())


def save_df_to_csv():
    app_train.to_csv("C:\\Users\\şerefcanmemiş\\Downloads\\home-credit-default-risk\\app_train.csv", header = True, index= False)
    app_test.to_csv("C:\\Users\\şerefcanmemiş\\Downloads\\home-credit-default-risk\\app_test.csv", header = True, index= False)


app_train = pd.read_csv("C:\\Users\\şerefcanmemiş\\Downloads\\home-credit-default-risk\\app_train.csv")
app_test = pd.read_csv("C:\\Users\\şerefcanmemiş\\Downloads\\home-credit-default-risk\\app_test.csv")

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

# Correlation between target and other features (age have positive impact and ext1,ext2,ext3 have negative impact)
corr = app_train.corr()['TARGET'].sort_values(ascending=True)
corr.tail(30)
corr.head(30)

# There are some mistakes about days employed bc 365243 can not be real employed days
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYED'].value_counts()
plt.hist(app_train['DAYS_EMPLOYED'])
plt.show()

# lets handle them
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]

print("anomalies default", anom['TARGET'].mean() * 100, "%")
print("non nomalies default", non_anom['TARGET'].mean() * 100, "%")
print(f"There are {len(anom)} anom days of employment in train set")

# Creating employed days anomaly flag
app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243
# anomalies returning to NA values
app_train['DAYS_EMPLOYED'].replace(365243,np.nan, inplace= True)

app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243
# anomalies returning to NA values
app_test['DAYS_EMPLOYED'].replace(365243,np.nan, inplace= True)
print('There are %d anom in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))


# Checking whether younger or older people are linked to late repayment of the loan
agg = app_train[['DAYS_BIRTH','TARGET','AMT_CREDIT']].copy()
agg['DAYS_BIRTH_YEARS'] = -app_train['DAYS_BIRTH'] / 365

# showing skewness of target 0 and 1, younger people more tend to target 1
sns.kdeplot(agg.loc[agg['TARGET'] == 1]['DAYS_BIRTH_YEARS'], color='red')
sns.kdeplot(agg.loc[agg['TARGET'] == 0]['DAYS_BIRTH_YEARS'], color='blue')
plt.show()

agg['DAYS_BIRTH_YEARS'] = pd.qcut(agg['DAYS_BIRTH_YEARS'], q = 8)
age_groups = agg.groupby('DAYS_BIRTH_YEARS').mean()
age_groups

# the higher the age, the closer the target gets to 0 (which mean can repay the loan in time)
plt.bar(x=age_groups.index.astype(str), height=age_groups['TARGET'].to_numpy())
plt.show()

ext_table = app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','TARGET','DAYS_BIRTH']]
# correlation table
ext_corr = ext_table.corr()
sns.heatmap(ext_corr, annot=True)
plt.show()

# lets look at ext1, ext2, ext2
for i, ext in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(ext_table.loc[ext_table['TARGET'] == 1][ext], color='red')
    sns.kdeplot(ext_table.loc[ext_table['TARGET'] == 0][ext], color='blue')
    plt.title(f'distribution of {ext} by Target')
plt.show()

app_test = app_test.drop('TARGET', axis = 1)

poly_features_train = app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']].copy()
poly_features_test =  app_test[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']].copy()

poly_target = poly_features_train['TARGET'].copy()
poly_features_train = poly_features_train.drop('TARGET',axis=1)
# imputing na values
def impute_na(poly_features_train, poly_features_test):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(poly_features_train)
    poly_train = imp.transform(poly_features_train)
    poly_test = imp.transform(poly_features_test)
    return poly_train, poly_test
poly_train, poly_test = impute_na(poly_features_train, poly_features_test)
poly_train.shape

# create Polynomial features
def create_poly_features(poly_train, poly_test):
    poly_transformer = PolynomialFeatures(degree=3)
    poly_train = poly_transformer.fit_transform(poly_train)
    poly_test = poly_transformer.transform(poly_test)
    poly_columns = poly_transformer.get_feature_names_out(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])

    poly_features_train = pd.DataFrame(data = poly_train, columns= poly_columns)
    poly_features_test = pd.DataFrame(data = poly_test, columns= poly_columns)
    print(poly_features_train.shape)
    print(poly_features_test.shape)
    return poly_features_train, poly_features_test
poly_train, poly_test = create_poly_features(poly_train, poly_test)
poly_train.shape

poly_train['TARGET'] = poly_target

# lets check out new features correlations
def poly_corr(poly_train):
    poly_corr = poly_train.corr()['TARGET'].sort_values()
    print(poly_corr)
poly_corr(poly_train)
poly_train.drop('1', axis=1, inplace=True)

# Left join poly and app data
def leftjoin_poly_app_data(poly_train, poly_test, app_train, app_test, target):
    poly_train['SK_ID_CURR'] = app_train['SK_ID_CURR']
    poly_test['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_train, on='SK_ID_CURR', how= 'left')
    app_test_poly = app_test.merge(poly_test, on='SK_ID_CURR', how= 'left')
    print("Poly train shape:",app_train_poly.shape)
    print("Poly test shape", app_test_poly.shape)

    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis = 1)
    app_train_poly['TARGET'] = target
    print("Poly train shape:",app_train_poly.shape)
    print("Poly test shape", app_test_poly.shape)
    return app_train_poly, app_test_poly
app_train_poly, app_test_poly = leftjoin_poly_app_data(poly_train, poly_test, app_train, app_test, target=poly_target)

app_train_poly = app_train_poly.dropna(subset=['TARGET'])

# Domain features
app_train.columns[:20]
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

def create_domain_df(app_train_domain, app_test_domain):
    app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / -app_train_domain['DAYS_BIRTH']
    app_train_domain['INCOME_PER_PERSON_FAMILY'] = app_train_domain['AMT_INCOME_TOTAL'] / (app_train_domain['CNT_CHILDREN'] + 1)
    app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

    app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / -app_test_domain['DAYS_BIRTH']
    app_test_domain['INCOME_PER_PERSON_FAMILY'] = app_test_domain['AMT_INCOME_TOTAL'] / (app_test_domain['CNT_CHILDREN'] + 1)
    app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
    return app_train_domain, app_test_domain

app_train_domain, app_test_domain = create_domain_df(app_train_domain, app_test_domain)


# Let continue with Modelling
train = app_train.copy()
test = app_test.copy()

def drop_target(train):
    if 'TARGET' in train.columns:
        train_target = train['TARGET']
        train.drop('TARGET',axis = 1, inplace= True)
    else:
        print('train already does not have TARGET column')
    return train_target
    
train_target = drop_target(train = train)

# Preprocessing
def drop_id(df):
    if 'SK_ID_CURR' in df.columns:
        df.drop('SK_ID_CURR',axis = 1, inplace= True)
    else:
        print('df already does not have SK_ID_CURR')
drop_id(train)
drop_id(test)
    
def pipeline(train, test):
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, train.columns.to_list())
    ])
    train = preprocessor.fit_transform(train)
    test = preprocessor.transform(test)
    return train, test
train, test = pipeline(train, test)

# Logistic Regression
def logistic_regression_results(train, test, train_target):
    lr = LogisticRegression(max_iter=3000)
    lr.fit(train,train_target)
    proba = lr.predict_proba(test)[:,1]

    # cross validation
    cvl = cross_val_score(lr, train, train_target, cv=5, scoring='roc_auc', n_jobs=-1) # around 0.74 - 0.75
    print("predict_proba results:", proba)
    print("cross validation score roc_auc:", cvl)
logistic_regression_results(train, test, train_target)


# Random Forest
train = app_train.copy()
test = app_test.copy()

# preprocessing
train_target = drop_target(train)
drop_id(train)
drop_id(test)

# no need to scale in ensemble models
train, test = pipeline(train, test)

def a():
    input('bir sayi giriniz')
    print('Celine aşığım')
a()

def random_forest_results(train, test, train_target):
    rfc = RandomForestClassifier()
    rfc.fit(train, train_target)
    proba = rfc.predict_proba(test)[:,1]

    # cross validation
    cvl = cross_val_score(rfc, train, train_target, cv=5, scoring='roc_auc', n_jobs=-1) 
    print("predict_proba results:", proba)
    print("cross validation score roc_auc:", cvl)
random_forest_results(train, test, train_target) # cv around 0.74 - 0.75


# Random Forest for polynomial features
train = app_train_poly.copy()
test = app_test_poly.copy()

# preprocessing
train_target = drop_target(train)

drop_id(train)
drop_id(test)

train, test = pipeline(train, test)
random_forest_results(train, test, train_target) # cv around 0.70 - 71

# Random Forest for domain features
train = app_train_domain.copy()
test = app_test_domain.copy()

# Preprocessing
train_target = drop_target(train)
drop_id(train)
drop_id(test)

train, test = pipeline(train, test)

random_forest_results(train, test, train_target)

