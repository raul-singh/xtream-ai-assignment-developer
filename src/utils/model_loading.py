import logging
import os
import pickle
from typing import Any

import pandas as pd

# Create and initialize logger
logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    pass


def load_model_report(report_dir: str) -> pd.DataFrame:
    report_path = os.path.join(
        report_dir,
        "report.csv"
    )

    try:
        report_df = pd.read_csv(report_path)
        logger.info("Model report loaded.")

    except FileNotFoundError:
        raise FileNotFoundError(
            "No model report found. There are no trained models."
        )

    return report_df


def get_best_model_id(
    model_type: str,
    report_df: pd.DataFrame,
    criteria: str = "mae"
):
    if model_type != "best":
        report_df = report_df.loc[report_df["type"] == model_type]

    if criteria == "mae":
        ascending = True
    elif criteria == "r2":
        ascending = False

    ranking = report_df[criteria].sort_values(ascending=ascending).index

    if len(ranking) == 0:
        raise ModelNotFoundError

    best = report_df.loc[ranking[0], "model_id"]
    pipeline_path = report_df.loc[ranking[0], "pipeline_path"]

    logger.info(
        "Found best %s model with %s of %f",
        best,
        criteria,
        report_df.loc[ranking[0], criteria]
    )

    return best, pipeline_path


def load_model(
    model_id: str,
    model_dir_path: str
) -> Any:

    with open(
        os.path.join(model_dir_path, "model_files", model_id),
        "rb"
    ) as input_file:
        model = pickle.load(input_file)

    logger.info("Model loaded.")

    return model
