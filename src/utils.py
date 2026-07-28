import os
import sys

import numpy as np
import pandas as pd
import pickle
from src.logger import logging
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(xtrain,ytrain)

            # Predict Training data
            y_train_pred = model.predict(xtrain)

            # Predict Testing data
            y_test_pred =model.predict(xtest)

            # Get R2 scores for train and test data
            train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)


def model_metrics(true, predicted):
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e, sys)