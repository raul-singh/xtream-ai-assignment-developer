import argparse
import logging

from fastapi import FastAPI, Request
import uvicorn
from assignment.constants.constants import DATASET_PATH, MODEL_DIR_PATH
from assignment.database.database import db_check, store_request

from assignment.utils.utils import (
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


app = FastAPI()


@app.get("/n-similar")
def get_similar_diamonds(
    cut: str,
    color: str,
    clarity: str,
    carat: float,
    n: int,
    request: Request
) -> list:

    global diamond_df

    similar_diamonds = diamond_df.loc[
        (diamond_df["cut"] == cut)
        & (diamond_df["color"] == color)
        & (diamond_df["clarity"] == clarity)
    ]

    ranked_similarity = abs(similar_diamonds["carat"] - carat).sort_values()
    selection = ranked_similarity.index.to_list()[:n]
    response = diamond_df.loc[selection].to_dict("records")
    store_request(request, response, db_url, db_name, collection_name)

    return response


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
    request: Request,
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

    prediction = float(model.predict(to_predict)[0])
    store_request(request, prediction, db_url, db_name, collection_name)

    return prediction


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
        default=MODEL_DIR_PATH,
        help="specify different directory where report and model are saved",
    )

    parser.add_argument(
        "--db-url",
        type=str,
        nargs='?',
        default="mongodb://localhost:27017/",
        help="the mongodb database url",
    )

    parser.add_argument(
        "--db-name",
        type=str,
        nargs='?',
        default="diamond_db",
        help="the mongodb database name",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        nargs='?',
        default="api_requests",
        help="the mongodb collection name",
    )

    args = parser.parse_args()
    model_to_use = args.model
    criteria = args.criteria
    dataset_path = args.dataset
    save_dir = args.save_dir

    global db_url, db_name, collection_name

    db_url = args.db_url
    db_name = args.db_name
    collection_name = args.collection_name

    db_check(db_url, db_name, collection_name)

    global diamond_df, model, model_type

    diamond_df = load_dataset(dataset_path)
    model, model_type = load_best_model(
        save_dir,
        None if model_to_use == "best" else model_to_use,
        criteria
        )

    uvicorn.run(app)
