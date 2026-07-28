import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            logging.info("Creating preprocessing object")

            #####################################################
            # Feature Groups
            #####################################################

            skewed_features = [
                "Study_Hours"
            ]

            numeric_features = [
                "Age",
                "Avg_Daily_Usage_Hours",
                "Daily_Unlocks",
                "Physical_Activity_Hours",
                "Sleep_Hours_Per_Night",
            ]

            ordinal_features = [
                "Stress_Level"
            ]

            nominal_features = [
                "Gender",
                "Academic_Level",
                "Most_Used_Platform",
                "Purpose_Of_Use",
                "Country",
            ]

            #####################################################
            # Skewed Pipeline
            #####################################################

            skew_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "log_transform",
                        FunctionTransformer(np.log1p)
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            #####################################################
            # Numeric Pipeline
            #####################################################

            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            #####################################################
            # Ordinal Pipeline
            #####################################################

            ordinal_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "Low",
                                    "Medium",
                                    "High",
                                    "Very High",
                                ]
                            ]
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            #####################################################
            # Nominal Pipeline
            #####################################################

            nominal_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            #####################################################
            # Column Transformer
            #####################################################

            preprocessor = ColumnTransformer(
                transformers=[

                    (
                        "Skewed",
                        skew_pipeline,
                        skewed_features,
                    ),

                    (
                        "Numeric",
                        numeric_pipeline,
                        numeric_features,
                    ),

                    (
                        "Ordinal",
                        ordinal_pipeline,
                        ordinal_features,
                    ),

                    (
                        "Nominal",
                        nominal_pipeline,
                        nominal_features,
                    ),
                ]
            )

            logging.info("Preprocessor created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    ##############################################################
    # Data Transformation
    ##############################################################

    def initiate_data_transformation(
        self,
        train_path,
        test_path,
    ):

        try:

            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train Shape : {train_df.shape}")
            logging.info(f"Test Shape : {test_df.shape}")

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            target_column = "Mental_Health_Score"

            X_train = train_df.drop(
                columns=[target_column],
                axis=1,
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=[target_column],
                axis=1,
            )

            y_test = test_df[target_column]

            logging.info("Applying preprocessing")

            X_train_arr = preprocessing_obj.fit_transform(
                X_train
            )

            X_test_arr = preprocessing_obj.transform(
                X_test
            )

            train_arr = np.c_[
                X_train_arr,
                np.array(y_train),
            ]

            test_arr = np.c_[
                X_test_arr,
                np.array(y_test),
            ]

            logging.info("Saving preprocessor")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Data Transformation Completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e, sys)