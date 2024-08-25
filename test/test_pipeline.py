import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from src import pipeline


@pytest.mark.order(1)
@pytest.mark.parametrize("pipeline_file", [
    "linear.yml",
    "xgboost_tuned.yml",
    "xgboost.yml"
])
def test_pipeline(pipeline_file: str):

    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")

    model_save_path = os.path.join("test", "test_models")
    trained_models_path = os.path.join(model_save_path, "model_files")
    os.makedirs(trained_models_path, exist_ok=True)
    report_path = os.path.join(model_save_path, "report.csv")
    pipeline_dir = os.path.join("test", "test_pipelines")

    pipeline_path = os.path.join(pipeline_dir, pipeline_file)

    n_trained_models = len(os.listdir(trained_models_path))

    try:
        n_models_report = len(pd.read_csv(report_path))
    except FileNotFoundError:
        n_models_report = 0

    pipeline(pipeline_path, dataset_path, model_save_path)

    n_trained_models_after = len(os.listdir(trained_models_path))
    n_models_report_after = len(pd.read_csv(report_path))

    assert n_models_report_after - n_models_report == 1
    assert n_trained_models_after - n_trained_models == 1
