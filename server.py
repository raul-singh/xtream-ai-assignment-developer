import logging
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException

from src.server.database import db_check, store_request
from src.server.server_logic import (
    make_prediction,
    similar_diamond_request,
)
from src.utils.model_loading import ModelNotFoundError
import argparse

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(name)s %(levelname)s: %(message)s",
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

    try:
        prediction = make_prediction(model, sample, criteria)
        store_request(request, prediction)
        return prediction

    except ModelNotFoundError:
        if model == "best":
            raise HTTPException(
                status_code=404,
                detail="No trained models found in the server."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No {model} trained model found."
            )

    except FileNotFoundError:
        raise HTTPException(
                status_code=404,
                detail="No trained models found in the server."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-path",
        nargs='?',
        default="",
        help=(
            "specify path of a custom environment file, "
            "mainly for testing purposes"
        )
    )
    args = parser.parse_args()
    custom_env = args.env_path

    if custom_env == "":
        load_dotenv()
    else:
        load_dotenv(custom_env, override=True)

    db_check()
    uvicorn.run(app)


if __name__ == "__main__":
    main()
