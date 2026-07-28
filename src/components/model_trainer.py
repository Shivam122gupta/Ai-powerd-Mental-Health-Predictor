# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import model_metrics

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
            }
            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for Rnadom Forest')

            # Hyperparameter tuning on Random Forest
            # Initializing catboost
            lf = RandomForestRegressor()
            # Creating the hyperparameter grid
            param_grid = {
                'n_estimators' : [100,200,300],
                'max_depth'    : [5,10,15],
                'min_samples_split' : [2,5,10],
                'min_samples_leaf' : [1,2,4]
                }
            random_search = RandomizedSearchCV(
                    estimator=lf,
                    param_distributions=param_grid,
                    n_iter= 15 ,
                    cv = 5,
                    scoring='r2',
                    random_state=42,
                    n_jobs=-1 # processor free h use parallel mein laga do
                )
            # Fit the model
            random_search.fit(xtrain, ytrain)
             # Print the tuned parameters and score
            print(f'Best Catboost parameters : {random_search.best_params_}')
            print(f'Best Catboost Score : {random_search.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = random_search.best_estimator_
            logging.info('Hyperparameter tuning complete for Rnadom Forest')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = random_search
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            ytest_pred = best_cbr.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest, ytest_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2 
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)