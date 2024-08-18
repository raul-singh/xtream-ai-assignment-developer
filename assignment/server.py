import argparse
import logging
import os

from fastapi import FastAPI
import uvicorn
from training_pipeline import DATASET_PATH, DIR_PATH

from utils import (
    load_best_model,
    load_dataset,
    preprocess_linear_sample,
    preprocess_xgboost_sample,
)

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)

# Change working directory so that code so that the behaviour
# is the same even if the code is executed from a different location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = FastAPI()


@app.get("/n-similar")
def get_similar_diamonds(
    cut: str,
    color: str,
    clarity: str,
    carat: float,
    n: int,
):

    global diamond_df

    similar_diamonds = diamond_df.loc[
        (diamond_df["cut"] == cut)
        & (diamond_df["color"] == color)
        & (diamond_df["clarity"] == clarity)
    ]

    ranked_similarity = abs(similar_diamonds["carat"] - carat).sort_values()
    selection = ranked_similarity.index.to_list()[:n]
    return diamond_df.loc[selection].to_dict("records")


@app.get("/prediction")
def predict(
    carat: float,
    cut: str,
    color: str,
    clarity: str,
    depth: float,
    table: float,
    x: float,
    y: float,
    z: float,
) -> float:

    global diamond_df, model_type

    sample = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }

    if model_type == "linear":
        to_predict = preprocess_linear_sample(sample)
    elif model_type == "xgboost":
        to_predict = preprocess_xgboost_sample(sample)

    p = model.predict(to_predict)[0]
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["linear", "xgboost", "best"],
        nargs='?',
        default="best",
        help="specify which type of model to use, or to just pick the best one"
    )

    parser.add_argument(
        "-c",
        "--criteria",
        type=str,
        choices=["MAE", "r2"],
        nargs='?',
        default="MAE",
        help="which criteria to use to pick the best model"
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        nargs='?',
        default=DATASET_PATH,
        help="the dataset path/url",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        nargs='?',
        default=DIR_PATH,
        help="specify different directory where report and model are saved",
    )

    args = parser.parse_args()
    model_to_use = args.model
    criteria = args.criteria
    dataset_path = args.dataset
    save_dir = args.save_dir

    global diamond_df, model, model_type

    diamond_df = load_dataset(dataset_path)

    model, model_type = load_best_model(
        save_dir,
        None if model_to_use == "best" else model_to_use,
        criteria
        )

    uvicorn.run(app)
