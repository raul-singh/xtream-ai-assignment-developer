import importlib
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Union

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from src.pipeline.pipeline_config import load_config

from src.preprocessing.train_preprocess import (
    get_train_test_dataset,
    preprocess_data,
)
from src.utils.utils import extract_args_kwargs, load_dataset

DEFAULT_TRAIN_TEST_SPLIT = 0.2

# Create and initialize logger
logger = logging.getLogger(__name__)


def train_evaluate_model(
    train_dataset: tuple[Any, Any],
    test_dataset: tuple[Any, Any],
    model_class: Any,
    model_type: str,
    pipeline_path: str,
    save_dir_path: str,
    **model_kwargs
):

    x_train, y_train = train_dataset
    x_test, y_test = test_dataset

    model = model_class(**model_kwargs)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    r2 = round(r2_score(y_test, pred), 4)
    mae = round(mean_absolute_error(y_test, pred), 2)

    logger.info("%s trained. R2 = %f, MAE = %f", model_type, r2, mae)

    # Save model as pickle
    model_file_path = os.path.join(save_dir_path, "model_files")
    os.makedirs(model_file_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_id = f"{timestamp}_{model_type}.pkl"

    # If we train two models in less of one second, we
    # may have a conflicting filename
    if model_id in os.listdir(model_file_path):
        idx = 2
        model_id = model_id.split(".")[0]
        while f"{model_id}-{idx}.pkl" in os.listdir(model_file_path):
            idx += 1
        model_id += f"-{idx}.pkl"

    filename = os.path.join(model_file_path, model_id)

    with open(filename, "wb") as file:
        pickle.dump(model, file)
    logger.info("Model file saved as %s", filename)

    report = {
        "type": model_type,
        "model_id": model_id,
        "r2": r2,
        "mae": mae,
        "pipeline_path": pipeline_path
    }

    # Save model performance report
    try:
        report_df = pd.read_csv(os.path.join(save_dir_path, "report.csv"))

    except FileNotFoundError:
        report_df = pd.DataFrame(
            [],
            columns=["type", "model_id", "r2", "mae", "pipeline_path"]
        )

    report["index"] = len(report_df)
    report_df.loc[len(report_df)] = report
    report_df.to_csv(os.path.join(save_dir_path, "report.csv"), index=False)
    logger.info(
        "Model report added to %s",
        os.path.join(save_dir_path, "report.csv")
    )


def tune_hyperparameters(
    train_dataset: tuple[Any, Any],
    pipeline_config: dict,
    model_class: Any
) -> dict[str, Union[int, float]]:

    x_train_ds, y_train_ds = train_dataset
    hyperparameter_tuning_config = pipeline_config["hyperparameter_tuning"]
    n_trials = hyperparameter_tuning_config["n_trials"]

    logger.info(
                "Performing hyperparameter tuning (%d trials)",
                n_trials
            )

    def objective(trial: optuna.trial.Trial) -> float:
        # Define hyperparameters to tune
        param = {}
        for k, v in hyperparameter_tuning_config["params"].items():
            if isinstance(v, dict):
                t = getattr(trial, v["trial"])
                args, kwargs = extract_args_kwargs(v["args"])
                param[k] = t(*args, **kwargs)

            else:
                param[k] = v

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_ds,
            y_train_ds,
            test_size=hyperparameter_tuning_config.get("test_split", 0.2),
            random_state=pipeline_config["seed"]
        )

        # Train the model
        model = model_class(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae

    study = optuna.create_study(
        direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def pipeline(
    pipeline_path: str,
    dataset_path: str,
    save_dir_path: str,
):
    diamond_df = load_dataset(dataset_path)
    pipeline_config = load_config(pipeline_path)
    x, y = preprocess_data(diamond_df, pipeline_config)
    train_dataset, test_dataset = get_train_test_dataset(
        x,
        y,
        pipeline_config["pipeline"].get(
            "test_split",
            DEFAULT_TRAIN_TEST_SPLIT
        ),
        pipeline_config["seed"]
    )

    model_module = importlib.import_module(pipeline_config["module"])
    model_class = getattr(model_module, pipeline_config["class"])
    name = pipeline_config["name"]

    model_kwargs = pipeline_config.get("model_kwargs", {})
    hyperparameter_tuning = pipeline_config.get("hyperparameter_tuning", None)

    if hyperparameter_tuning is not None:
        model_kwargs.update(
            tune_hyperparameters(
                train_dataset,
                pipeline_config,
                model_class,
            )
        )

    logger.info("Training %s model...", name)
    train_evaluate_model(
        train_dataset,
        test_dataset,
        model_class,
        name,
        pipeline_path,
        save_dir_path,
        **model_kwargs
    )
