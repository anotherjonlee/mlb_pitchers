from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
import creds
from pathlib import Path
from sklearn.model_selection import train_test_split
from os import system

class pitchers(object):
    def __init__(self):
        self.pw = creds.mysql_pw
        self.host = 'localhost:3306'
        self.database = 'lahmansbaseballdb'
        self.user = 'root'
        self.query = None
        self.file_name = None
        self.sql_path = '../data/lahman-mysql-dump.sql'
        self.command = """mysql -u %s -p %s --host 'localhost' --port 3306 < %s""" %(self.user, self.pw, self.sql_path)
        system(self.command)
        print('SQL database has been created.')
        self.engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}/{self.database}',pool_recycle=3600)
        self.connection = self.engine.connect()
        self.connection_status = None

    def statistical_analysis(self, query, file_name, connection_status):
        d = Path().resolve().parent
        print('Current path is: ', d)
        file_path = str(d) + '/data/' + file_name + '.csv'
        
        df = pd.read_sql(query, self.connection)

        df.dropna(inplace=True)
        df.to_csv(file_path,header=True,index=False)

        print('CSV files have been created from a SQL query for statistical analysis.')
        
        if connection_status == 'y':
            self.connection.close()
            self.engine.dispose()

    def sagemaker(self, query, file_name, connection_status):
        np.random.seed(10)
        d = Path().resolve().parent
        print('path is: ', d)
        file_path = str(d) + '/data/' + str(file_name) + '.csv'
        train_path = str(d) + '/data/training_dataset.csv'
        validation_path = str(d) + '/data/validation_dataset.csv'
        
        df = pd.read_sql(query, self.connection)

        # One hot encode categorical columns
        one_hot_cols = ['playerID','teamID','lgID', 'throws']
        dummies = pd.get_dummies(df[one_hot_cols], prefix=['player','team','league','throws'])
        
        df = df.join(dummies)
        df.drop(one_hot_cols,axis=1,inplace=True)
        df.dropna(inplace=True)
        df.to_csv(file_path,header=True,index=False)

        # Train test split the dataframe for Sagemaker 
        training_dataset,validation_dataset = train_test_split(df,test_size=0.1)
        
        training_dataset.to_csv(train_path, index=False, header=False)
        validation_dataset.to_csv(validation_path, index=False, header=False)

        print('CSV files have been created from a SQL query for SageMaker.')
        if connection_status == 'y':
            self.connection.close()
            self.engine.dispose()

if __name__ == '__main__':
    mlb = pitchers()
    
    query = """
        WITH lg_averages (yearID, lgID, lg_era, lg_hr, lg_bb, lg_hbp, lg_so, lg_ip) AS (
            SELECT
                yearID,
                lgID,
                AVG(ERA),
                AVG(HR),
                AVG(BB),
                AVG(HBP),
                AVG(SO),
                ROUND(AVG((IPouts/3)),2)
            FROM 
                pitching
            GROUP BY 
                1,2
        )
        SELECT
            p.*,
            people.throws,
            ROUND(whip,2) whip,
            ROUND(fip,2) fip
        FROM 
            pitching p 
        JOIN 
            people ON people.playerID = p.playerID
        JOIN 
            salaries s ON s.playerID = p.playerID AND s.yearID = p.yearID AND s.teamID = p.teamID
        JOIN 
            lg_averages l ON p.lgID = l.lgID AND p.yearID  = l.yearID,
        LATERAL (
            SELECT (p.BB + p.H) / (p.IPouts / 3)
        ) a(whip),
        LATERAL(
            SELECT (13 * p.HR + 3 * (p.BB + p.HBP) - (2 * p.SO)) / (p.IPouts / 3)
        ) b(fip_less),
        LATERAL(
            SELECT
                lg_era - ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_so)) / lg_ip
        ) d(fip_constant),
        LATERAL (
            SELECT fip_less + fip_constant
        ) e(fip)
        ORDER BY p.yearID 
    """
    mlb.statistical_analysis(query,'performances','n')

    query = """
        SELECT
            s.salary,
            p.*,
            people.throws
        FROM
            pitching p
            JOIN people ON people.playerID = p.playerID
            JOIN salaries s ON p.playerID = s.playerID
                AND p.yearID = s.yearID
                AND p.teamID = s.teamID
            ORDER BY
                p.yearID,
                p.playerID
    """
    mlb.sagemaker(query,'salaries','y')
