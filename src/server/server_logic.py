import logging
import os
from typing import Any

from src.preprocessing.inference_preprocess import preprocess_sample
from src.utils.model_loading import (
    get_best_model_id,
    load_model,
    load_model_report,
)
from src.utils.utils import (
    load_dataset,
)

# Create and initialize logger
logger = logging.getLogger(__name__)

# These variables are initialized to None because
# they are lazily loaded when needed
diamond_df = None
loaded_model = None
model_id = None
loaded_model_pipeline_path = None
report_df = None


def similar_diamond_request(payload: dict[str, Any]) -> list[dict[str: Any]]:

    global diamond_df

    if diamond_df is None:
        dataset_path = os.getenv("DATASET_PATH")
        diamond_df = load_dataset(dataset_path)

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
    criteria: str,
) -> float:

    global diamond_df, report_df, loaded_model, loaded_model_pipeline_path

    model_dir_path = os.getenv("MODEL_DIR_PATH")
    logger.info("Model directory path is %s", model_dir_path)

    if report_df is None:
        report_df = load_model_report(model_dir_path)

    best_model_id, pipeline_path = get_best_model_id(
        model_type,
        report_df,
        criteria
    )

    if best_model_id != model_id:
        loaded_model = load_model(best_model_id, model_dir_path)
        loaded_model_pipeline_path = pipeline_path
    else:
        logger.info("Model is already loaded.")

    to_predict = preprocess_sample(
        sample,
        diamond_df,
        loaded_model_pipeline_path
    )

    return float(loaded_model.predict(to_predict)[0])
