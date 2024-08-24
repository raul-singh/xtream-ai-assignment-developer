import os
import shutil
from dotenv import load_dotenv
import pymongo
import requests

load_dotenv()
API_SERVER_URL = os.getenv("API_SERVER_URL")
DB_URL = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME")
DB_COLLECTION = os.getenv("DB_COLLECTION")


def test_similar_diamonds():
    payload = {
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI1",
        "carat": 0.7,
        "n": 5
    }

    with pymongo.MongoClient(DB_URL) as client:
        col = client[DB_NAME][DB_COLLECTION]
        n_docs_before = len(
            list(col.find({"path": "/similar"}))
        )

        response = requests.post(
            f"http://{API_SERVER_URL}/similar",
            params=payload
        )

        n_docs_after = len(
            list(col.find({"path": "/similar"}))
        )

    output = response.json()

    assert response.status_code == 200
    assert isinstance(output, list)
    assert len(output) <= payload["n"]
    assert n_docs_after - n_docs_before == 1


def test_diamond_prediction():
    payload = {
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI1",
        "carat": 0.7,
        "depth": 61.0,
        "table": 56,
        "x": 5.74,
        "y": 5.76,
        "z": 3.51,
        "model": "linear",
        "criteria": "mae"
    }

    with pymongo.MongoClient(DB_URL) as client:
        col = client[DB_NAME][DB_COLLECTION]
        n_docs_before = len(
            list(col.find({"path": "/prediction"}))
        )

        response = requests.post(
            f"http://{API_SERVER_URL}/prediction",
            params=payload
        )

        n_docs_after = len(
            list(col.find({"path": "/prediction"}))
        )

    output = response.json()

    assert response.status_code == 200
    assert isinstance(output, float)
    assert n_docs_after - n_docs_before == 1


def test_prediction_on_absent_model():
    payload = {
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI1",
        "carat": 0.7,
        "depth": 61.0,
        "table": 56,
        "x": 5.74,
        "y": 5.76,
        "z": 3.51,
        "model": "a_model_type_not_present",
        "criteria": "mae"
    }

    with pymongo.MongoClient(DB_URL) as client:
        col = client[DB_NAME][DB_COLLECTION]
        n_docs_before = len(
            list(col.find({"path": "/prediction"}))
        )

        response = requests.post(
            f"http://{API_SERVER_URL}/prediction",
            params=payload
        )

        n_docs_after = len(
            list(col.find({"path": "/prediction"}))
        )

    assert response.status_code == 404
    assert n_docs_after == n_docs_before


def test_prediction_no_report():
    payload = {
        "cut": "Ideal",
        "color": "H",
        "clarity": "SI1",
        "carat": 0.7,
        "depth": 61.0,
        "table": 56,
        "x": 5.74,
        "y": 5.76,
        "z": 3.51,
    }

    shutil.rmtree(os.path.join("test", "test_models"))

    with pymongo.MongoClient(DB_URL) as client:
        col = client[DB_NAME][DB_COLLECTION]
        n_docs_before = len(
            list(col.find({"path": "/prediction"}))
        )

        response = requests.post(
            f"http://{API_SERVER_URL}/prediction",
            params=payload
        )

        n_docs_after = len(
            list(col.find({"path": "/prediction"}))
        )

    assert response.status_code == 404
    assert n_docs_after == n_docs_before
