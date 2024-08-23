import logging

import uvicorn
from fastapi import FastAPI, Request

from assignment.server.database import db_check, store_request
from assignment.server.server_logic import (
    make_prediction,
    similar_diamond_request,
)

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


def main():
    db_check()
    uvicorn.run(app)


if __name__ == "__main__":
    main()
