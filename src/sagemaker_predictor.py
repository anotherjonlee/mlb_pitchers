import os
# os.system("exec data_cleaner.py") # Executes the data_cleaner.py to prepare dataframes
import pandas as pd
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.predictor import csv_serializer, json_deserializer
from time import strftime,gmtime
import sys
from pathlib import Path
import creds

class sm_predictor(object):
    
    def __init__(self):
        self.sess = sagemaker.Session()
        self.bucket = self.sess.default_bucket()
        self.role = creds.aws_role
        self.region = boto3.Session().region_name
        self.container = image_uris.retrieve('linear-learner', self.region)
        self.prefix = 'mlb-salary-prediction'
        
    def data_prep(self):
        d = Path().resolve().parent
        print('Current path is: ', d)

        training_data_path = self.sess.upload_data(path= str(d) + '/data/training_dataset.csv', key_prefix=self.prefix + '/input/training')
        validation_data_path = self.sess.upload_data(path= str(d) + '/data/validation_dataset.csv', key_prefix=self.prefix + '/input/validation')
        
        print('S3 path for training dataset:',training_data_path)
        print('S3 path for validation dataset:',validation_data_path)

        training_data_channel = sagemaker.TrainingInput(s3_data=training_data_path,content_type='text/csv')
        validation_data_channel = sagemaker.TrainingInput(s3_data=validation_data_path,content_type='text/csv')

        print('Training data channel has been set up.')
        print('Validation data channel has been set up.')

        data_channels = {'train': training_data_channel,'validation':validation_data_channel}
        return data_channels

    def estimator(self, batch_n):
        ll_estimator = Estimator(
            self.container,
            role = self.role,
            instance_count=1,
            instance_type='ml.m5.large',
            output_path = 's3://{}/{}/output'.format(self.bucket,self.prefix)
        )

        ll_estimator.set_hyperparameters(predictor_type='regressor', mini_batch_size=batch_n)
        
        return ll_estimator
    
if __name__ == '__main__':
    sm = sm_predictor()
    data_channels = sm.data_prep()
    predictor = sm.estimator(32)
    predictor.fit(data_channels)
    print('Model has been fitted.')

