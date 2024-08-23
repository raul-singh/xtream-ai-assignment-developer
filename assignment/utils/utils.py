import logging
import os
import pickle
from typing import Union
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from assignment.preprocessing.train_preprocess import basic_preprocess_diamonds


# Create and initialize logger
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    logger.info("Loading dataset...")
    diamond_df = pd.read_csv(dataset_path)
    diamond_df = basic_preprocess_diamonds(diamond_df)
    logger.info("Dataset loaded.")
    return diamond_df


def load_model_report() -> pd.DataFrame:
    report_path = os.path.join(
        os.getenv("MODEL_DIR_PATH"),
        "report.csv"
    )

    try:
        report_df = pd.read_csv(report_path)

    except FileNotFoundError:
        raise FileNotFoundError((
            "No model report found. Make sure to train a model "
            "and put it in the right directory using training_pipeline.py"
        ))

    return report_df


def get_best_model_id(
    model_type: str,
    report_df: pd.DataFrame,
    criteria: str = "MAE"
):
    if model_type != "overall":
        report_df = report_df.loc[report_df["type"] == model_type]

    if criteria == "MAE":
        ascending = True
    elif criteria == "r2":
        ascending = False

    ranking = report_df[criteria].sort_values(ascending=ascending).index
    best = report_df.loc[ranking[0], "model_id"]
    actual_model_type = report_df.loc[ranking[0], "type"]

    logger.info(
        "Found best %s model with %s of %f",
        best,
        criteria,
        report_df.loc[ranking[0], criteria]
    )

    return best, actual_model_type


def load_model(
    model_id: str,
    model_dir_path: str
) -> Union[LinearRegression, XGBRegressor]:

    with open(
        os.path.join(model_dir_path, "model_files", model_id),
        "rb"
    ) as input_file:
        model = pickle.load(input_file)

    logger.info("Model loaded.")

    return model
