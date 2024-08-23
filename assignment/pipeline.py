import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Union

import optuna
import pandas as pd
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from assignment.constants.constants import DATASET_PATH, MODEL_DIR_PATH
from assignment.preprocessing.train_preprocess import (
    basic_preprocess_diamonds,
    preprocess_diamond_linear_reg,
    preprocess_diamond_xgboost,
)

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)


def train_ev_model(
    train_dataset: tuple[Any, Any],
    test_dataset: tuple[Any, Any],
    model_class: Any,
    model_type: str,
    save_dir: str,
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
    model_file_path = os.path.join(save_dir, "model_files")
    os.makedirs(model_file_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_id = f"{timestamp}_{model_type}.pkl"
    filename = os.path.join(model_file_path, model_id)

    with open(filename, "wb") as file:
        pickle.dump(model, file)
    logger.info("Model file saved as %s", filename)

    report = {
        "type": model_type,
        "model_id": model_id,
        "r2": r2,
        "MAE": mae
    }

    # Save model performance report
    try:
        report_df = pd.read_csv(os.path.join(save_dir, "report.csv"))

    except FileNotFoundError:
        report_df = pd.DataFrame([], columns=["type", "model_id", "r2", "MAE"])

    report["index"] = len(report_df)
    report_df.loc[len(report_df)] = report
    report_df.to_csv(os.path.join(save_dir, "report.csv"), index=False)
    logger.info("Model report added to %s",
                os.path.join(save_dir, "report.csv"))


def tune_hyperparameters(
    train_dataset: tuple[Any, Any],
    seed: int,
    n_trials: int
) -> dict[str, Union[int, float]]:

    x_train_ds, y_train_ds = train_dataset

    def objective(trial: optuna.trial.Trial) -> float:
        # Define hyperparameters to tune
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical(
                'colsample_bytree', [0.3, 0.4, 0.5, 0.7]
            ),
            'subsample': trial.suggest_categorical(
                'subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-8, 1.0, log=True
            ),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        # Split the training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_ds, y_train_ds, test_size=0.2, random_state=seed)

        # Train the model
        model = xgboost.XGBRegressor(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae

    study = optuna.create_study(
        direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials=n_trials)
    print(study.best_params)
    return study.best_params


def pipeline(
    model: str,
    dataset_path: str,
    hyperparameter_tuning: bool,
    seed: int,
    n_tuning_trials: int,
    save_dir: int,
    test_split: float,
):
    # Load dataset and perform basic preprocess common to all models
    logger.info("Loading dataset...")
    diamond_df = pd.read_csv(dataset_path)
    diamond_df = basic_preprocess_diamonds(diamond_df)
    logger.info("Dataset loaded.")

    if model == "linear":
        train_dataset, test_dataset = preprocess_diamond_linear_reg(
            diamond_df,  test_split, seed)
        logger.info("Training %s model...", model)
        train_ev_model(
            train_dataset,
            test_dataset,
            LinearRegression,
            model,
            save_dir
        )

    elif model == "xgboost":
        train_dataset, test_dataset = preprocess_diamond_xgboost(
            diamond_df, test_split, seed)
        model_kwargs = {
            "random_state": seed,
            "enable_categorical": True
        }
        if hyperparameter_tuning:
            logger.info(
                "Performing hyperparameter tuning (%d trials)",
                n_tuning_trials
            )
            model_kwargs.update(
                tune_hyperparameters(
                    train_dataset,
                    seed,
                    n_tuning_trials
                )
            )

        logger.info("Training %s model...", model)
        train_ev_model(
            train_dataset,
            test_dataset,
            xgboost.XGBRegressor,
            model,
            save_dir,
            **model_kwargs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["linear", "xgboost"],
        nargs='?',
        default="linear",
        help="specify which model to train and save"
    )

    parser.add_argument(
        "-t",
        "--tuning",
        action="store_true",
        help="perform hyperparameter tuning, works only with xgboost"
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        nargs='?',
        default=42,
        help="random seed",
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
        "--tuning-trials",
        type=int,
        nargs='?',
        default=100,
        help="how many hyperparameter tuning trials to perform",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        nargs='?',
        default=MODEL_DIR_PATH,
        help="specify different directory for models and reports",
    )

    parser.add_argument(
        "--test-split",
        type=float,
        nargs='?',
        default=0.2,
        help="the test/training split ratio.",
    )

    args = parser.parse_args()
    model = args.model
    dataset_path = args.dataset
    seed = args.seed
    tuning = args.tuning
    tuning_trials = args.tuning_trials
    save_dir = args.save_dir
    test_split = args.test_split

    pipeline(
        model,
        dataset_path,
        tuning,
        seed,
        tuning_trials,
        save_dir,
        test_split
    )
