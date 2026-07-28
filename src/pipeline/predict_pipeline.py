"""
predict_pipeline.py
====================
Production-level prediction pipeline for the Mental Health Score Predictor.

Classes:
    PredictPipeline  – loads artifacts and runs inference.
    CustomData       – maps raw HTTP form fields to a typed DataFrame row.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

# ---------------------------------------------------------------------------
# Module-level logger (re-uses the project logger config set up in logger.py)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH       = os.path.join(BASE_DIR, "artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")


# ---------------------------------------------------------------------------
# PredictPipeline
# ---------------------------------------------------------------------------
class PredictPipeline:
    """
    Loads the trained model and preprocessor once and exposes a `predict`
    method that accepts a pandas DataFrame of raw feature values.
    """

    def __init__(self):
        self.model       = None
        self.preprocessor = None
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        """Load preprocessor and model from disk. Raises CustomException on failure."""
        try:
            logger.info("Loading preprocessor from: %s", PREPROCESSOR_PATH)
            self.preprocessor = load_object(PREPROCESSOR_PATH)

            logger.info("Loading model from: %s", MODEL_PATH)
            raw_model = load_object(MODEL_PATH)

            # The model was saved as a RandomizedSearchCV object; extract best estimator.
            if hasattr(raw_model, "best_estimator_"):
                self.model = raw_model.best_estimator_
                logger.info(
                    "Extracted best_estimator_ from RandomizedSearchCV: %s",
                    type(self.model).__name__,
                )
            else:
                self.model = raw_model
                logger.info("Model loaded: %s", type(self.model).__name__)

        except Exception as e:
            logger.exception("Failed to load artifacts.")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transform raw features and return model predictions.

        Args:
            features (pd.DataFrame): DataFrame whose columns match the
                training feature set (before preprocessing).

        Returns:
            np.ndarray: Array of predicted Mental Health Score values.
        """
        try:
            logger.info("Running preprocessing on input data (shape=%s).", features.shape)
            transformed = self.preprocessor.transform(features)

            logger.info("Running model inference.")
            predictions = self.model.predict(transformed)

            logger.info("Prediction complete: %s", predictions)
            return predictions

        except Exception as e:
            logger.exception("Prediction failed.")
            raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# CustomData  – form → DataFrame mapper
# ---------------------------------------------------------------------------
class CustomData:
    """
    Captures all raw form values submitted from home.html and exposes
    `get_data_as_dataframe()` to produce a properly typed single-row
    DataFrame ready for the prediction pipeline.

    Feature order and column names must exactly match those used
    during model training (see data_transformation.py).
    """

    # Ordinal mapping used for basic validation (not encoding – preprocessor does that)
    STRESS_LEVELS = ["Low", "Medium", "High", "Very High"]

    def __init__(
        self,
        age: float,
        gender: str,
        academic_level: str,
        country: str,
        avg_daily_usage_hours: float,
        daily_unlocks: int,
        most_used_platform: str,
        purpose_of_use: str,
        study_hours: float,
        sleep_hours_per_night: float,
        physical_activity_hours: float,
        stress_level: str,
    ):
        self.Age                    = float(age)
        self.Gender                 = str(gender)
        self.Academic_Level         = str(academic_level)
        self.Country                = str(country)
        self.Avg_Daily_Usage_Hours  = float(avg_daily_usage_hours)
        self.Daily_Unlocks          = int(daily_unlocks)
        self.Most_Used_Platform     = str(most_used_platform)
        self.Purpose_Of_Use         = str(purpose_of_use)
        self.Study_Hours            = float(study_hours)
        self.Sleep_Hours_Per_Night  = float(sleep_hours_per_night)
        self.Physical_Activity_Hours = float(physical_activity_hours)
        self.Stress_Level           = str(stress_level)

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Returns a single-row DataFrame whose column names and dtypes match
        the training feature set consumed by the ColumnTransformer.
        """
        try:
            data = {
                # Skewed feature
                "Study_Hours": [self.Study_Hours],

                # Numeric features
                "Age":                     [self.Age],
                "Avg_Daily_Usage_Hours":   [self.Avg_Daily_Usage_Hours],
                "Daily_Unlocks":           [self.Daily_Unlocks],
                "Physical_Activity_Hours": [self.Physical_Activity_Hours],
                "Sleep_Hours_Per_Night":   [self.Sleep_Hours_Per_Night],

                # Ordinal feature
                "Stress_Level": [self.Stress_Level],

                # Nominal features
                "Gender":              [self.Gender],
                "Academic_Level":      [self.Academic_Level],
                "Most_Used_Platform":  [self.Most_Used_Platform],
                "Purpose_Of_Use":      [self.Purpose_Of_Use],
                "Country":             [self.Country],
            }

            df = pd.DataFrame(data)
            logger.info("Input DataFrame created:\n%s", df.to_string())
            return df

        except Exception as e:
            logger.exception("Failed to build input DataFrame.")
            raise CustomException(e, sys)