import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


user = os.getenv('user')
password = os.getenv('password')
host = os.getenv('host')
database = os.getenv('database')
folder = os.getenv('folder')


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

engine = create_engine(
    f'mysql+pymysql://{user}:{password}@{host}:3306/{database}?charset=utf8mb4',
    connect_args={"local_infile": 1}
)
def create_table_columns():
    for file, table_name in files_to_tables.items():
        path = os.path.join(folder,file).replace("\\",'/')
        df = pd.read_csv(path, encoding="utf-8")
        df.head(0).to_sql(table_name, con = engine, index=False, if_exists="replace")


def load_data_to_tables():
    for file, table_name in files_to_tables.items():
        path = os.path.join(folder,file).replace('\\','/')
        print(path)
        print(f"loading file into {table_name} ...")
        with engine.begin() as conn:
            conn.execute(
                f"""
                LOAD DATA LOCAL INFILE '{path}'
                INTO TABLE {table_name}
                FIELDS TERMINATED BY ','
                ENCLOSED BY '"'
                LINES TERMINATED BY '\\n'
                IGNORE 1 ROWS;
                """
            )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE application_combined
        ADD PRIMARY KEY (SK_ID_CURR);
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE bureau
        ADD FOREIGN KEY (SK_ID_CURR) REFERENCES application_combined(SK_ID_CURR)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE bureau
        ADD PRIMARY KEY (SK_ID_BUREAU)
        """
    )
    print('bureau primary key added')
    
with engine.begin() as conn:
    conn.execute(
        """
        delete from bureau_balance
        where SK_ID_BUREAU not in (select distinct(SK_ID_BUREAU) from bureau)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE bureau_balance
        ADD FOREIGN KEY (SK_ID_BUREAU) REFERENCES bureau(SK_ID_BUREAU)
        """
    )
    print('bureau_balance foreign key added')

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE previous_application
        ADD FOREIGN KEY (SK_ID_CURR) REFERENCES application_combined(SK_ID_CURR)
        """
    )
    
with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE previous_application
        ADD PRIMARY KEY (SK_ID_PREV)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE pos_cash_balance
        ADD FOREIGN KEY (SK_ID_CURR) REFERENCES application_combined(SK_ID_CURR)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE credit_card_balance
        ADD FOREIGN KEY (SK_ID_CURR) REFERENCES application_combined(SK_ID_CURR)
        """
    )


with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE pos_cash_balance
        ADD FOREIGN KEY (SK_ID_PREV) REFERENCES previous_application(SK_ID_PREV)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE installments_payments
        ADD FOREIGN KEY (SK_ID_PREV) REFERENCES previous_application(SK_ID_PREV)
        """
    )

with engine.begin() as conn:
    conn.execute(
        """
        ALTER TABLE credit_card_balance
        ADD FOREIGN KEY (SK_ID_PREV) REFERENCES previous_application(SK_ID_PREV)
        """
    )