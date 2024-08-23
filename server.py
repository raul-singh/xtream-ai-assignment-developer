import argparse
import logging
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Request
import uvicorn
from assignment.database.database import db_check, store_request
from assignment.server_logic import make_prediction, similar_diamond_request

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)

app = FastAPI()


@app.post("/similar")
def get_similar_diamonds(
    cut: str,
    color: str,
    clarity: str,
    carat: float,
    n: int,
    request: Request
) -> list:

    payload = {
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "carat": carat,
        "n": n,
    }

    response = similar_diamond_request(payload)
    store_request(request, response)

    return response


@app.post("/prediction")
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
    model: str = "best",
    criteria: str = "mae",
) -> float:

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

    prediction = make_prediction(model, sample, criteria)
    store_request(request, prediction)

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

    # Loading variables from .env file
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    MODEL_DIR_PATH = os.getenv("MODEL_DIR_PATH")

    db_check()

    uvicorn.run(app)
