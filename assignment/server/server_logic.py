import logging
import os
from typing import Any

from dotenv import load_dotenv

from assignment.preprocessing.inference_preprocess import (
    preprocess_linear_sample,
    preprocess_xgboost_sample,
)
from assignment.utils.utils import (
    get_best_model_id,
    load_dataset,
    load_model,
    load_model_report,
)

# Create and initialize logger
logger = logging.getLogger(__name__)

# Loading variables from .env file
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
MODEL_DIR_PATH = os.getenv("MODEL_DIR_PATH")

diamond_df = load_dataset(DATASET_PATH)

# These variables are initialized to None because
# they are lazily loaded when needed
loaded_model = None
model_id = None
loaded_model_type = None
report_df = None


def similar_diamond_request(payload: dict[str, Any]) -> list[dict[str: Any]]:

    global diamond_df

    similar_diamonds = diamond_df.loc[
        (diamond_df["cut"] == payload["cut"])
        & (diamond_df["color"] == payload["color"])
        & (diamond_df["clarity"] == payload["clarity"])
    ]

    ranked_similarity = (
        abs(similar_diamonds["carat"] - payload["carat"])
        .sort_values()
    )
    selection = ranked_similarity.index.to_list()[:payload["n"]]
    return diamond_df.loc[selection].to_dict("records")


def make_prediction(
    model_type: str,
    sample: dict[str, Any],
    criteria: str
) -> float:

    global diamond_df, report_df, loaded_model, loaded_model_type

    if report_df is None:
        report_df = load_model_report()
        logger.info("Model report loaded.")

    best_model_id, actual_model_type = get_best_model_id(
        model_type,
        report_df,
        criteria
    )

    if best_model_id != model_id:
        loaded_model = load_model(best_model_id, MODEL_DIR_PATH)
        loaded_model_type = actual_model_type
    else:
        logger.info("Model is already loaded.")

    if loaded_model_type == "linear":
        to_predict = preprocess_linear_sample(sample, diamond_df)
    elif loaded_model_type == "xgboost":
        to_predict = preprocess_xgboost_sample(sample)

    return float(loaded_model.predict(to_predict)[0])
