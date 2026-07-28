"""
app.py
======
Production Flask application for the Mental Health Score Predictor.

Routes:
    GET  /            → Landing page (index.html)
    GET  /prediction  → Blank prediction form (home.html)
    POST /prediction  → Run inference and return result (home.html)
"""

import sys
import logging

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
application = Flask(__name__)
app = application

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict_datapoint():
    """
    GET  → Show blank form.
    POST → Validate form data, run prediction pipeline, show result.
    """
    if request.method == "GET":
        return render_template("home.html")

    # ── POST: collect form values ──────────────────────────────────────────
    try:
        data = CustomData(
            age=request.form.get("Age"),
            gender=request.form.get("Gender"),
            academic_level=request.form.get("Academic_Level"),
            country=request.form.get("Country"),
            avg_daily_usage_hours=request.form.get("Avg_Daily_Usage_Hours"),
            daily_unlocks=request.form.get("Daily_Unlocks"),
            most_used_platform=request.form.get("Most_Used_Platform"),
            purpose_of_use=request.form.get("Purpose_Of_Use"),
            study_hours=request.form.get("Study_Hours"),
            sleep_hours_per_night=request.form.get("Sleep_Hours_Per_Night"),
            physical_activity_hours=request.form.get("Physical_Activity_Hours"),
            stress_level=request.form.get("Stress_Level"),
        )

        features_df = data.get_data_as_dataframe()

        # ── Run prediction ─────────────────────────────────────────────────
        pipeline = PredictPipeline()
        result = pipeline.predict(features_df)

        # Round to 2 decimal places for display
        score = round(float(result[0]), 2)
        logger.info("Prediction successful: %s", score)

        return render_template(
            "home.html",
            prediction_text=str(score),
        )

    except CustomException as ce:
        logger.error("CustomException during prediction: %s", ce)
        return render_template(
            "home.html",
            prediction_text=f"Error: {ce}",
        )

    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)