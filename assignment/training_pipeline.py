from datetime import datetime
import pickle
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
import xgboost

DIR_PATH = "./models"
DATASET_PATH = "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"

def basic_preprocess_diamonds(diamond_df):
    return diamond_df[(diamond_df.x * diamond_df.y * diamond_df.z != 0) & (diamond_df.price > 0)]

def preprocess_diamond_linear_reg(diamond_df, test_size=0.2, seed=42):
    diamond_df = diamond_df.drop(columns=['depth', 'table', 'y', 'z'])
    diamond_df = pd.get_dummies(diamond_df, columns=['cut', 'color', 'clarity'], drop_first=True)

    x = diamond_df.drop(columns='price')
    y = diamond_df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    train_dataset = x_train, y_train
    test_dataset = x_test, y_test

    return train_dataset, test_dataset

def preprocess_diamond_xgboost(diamond_df, test_size=0.2, seed=42):
    diamond_df = diamond_df.copy()
    diamond_df['cut'] = pd.Categorical(
        diamond_df['cut'],
        categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
        ordered=True
        )
    diamond_df['color'] = pd.Categorical(
        diamond_df['color'],
        categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
        ordered=True
        )
    diamond_df['clarity'] = pd.Categorical(
        diamond_df['clarity'],
        categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
        ordered=True
        )

    x = diamond_df.drop(columns='price')
    y = diamond_df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    train_dataset = x_train, y_train
    test_dataset = x_test, y_test

    return train_dataset, test_dataset

def train_ev_model(train_dataset, test_dataset, model_class, model_type, save_dir, **model_kwargs):
    x_train, y_train = train_dataset
    x_test, y_test = test_dataset

    model = model_class(**model_kwargs)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    r2 = round(r2_score(y_test, pred), 4)
    mae = round(mean_absolute_error(y_test, pred), 2)

    model_file_path = os.path.join(save_dir, "model_files")
    os.makedirs(model_file_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_id = f"{timestamp}_{model_type}.pkl"
    filename = os.path.join(model_file_path, model_id)

    with open(filename, "wb") as file:
        pickle.dump(model, file)

    report = {
        "type": model_type,
        "model_id": model_id,
        "r2": r2,
        "MAE": mae
    }

    try:
        report_df = pd.read_csv(os.path.join(save_dir, "report.csv"))

    except FileNotFoundError:
        report_df = pd.DataFrame([], columns=["type", "model_id", "r2", "MAE"])

    report["index"] = len(report_df)
    report_df.loc[len(report_df)] = report
    report_df.to_csv(os.path.join(save_dir, "report.csv"), index=False)
    print("aha")


def tune_hyperparameters(train_dataset, seed, n_trials=100):
    x_train_ds, y_train_ds = train_dataset

    def objective(trial: optuna.trial.Trial):
        # Define hyperparameters to tune
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        # Split the training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train_ds, y_train_ds, test_size=0.2, random_state=seed)

        # Train the model
        model = xgboost.XGBRegressor(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae

    study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def pipeline(model, dataset_path, hyperparameter_tuning, seed, n_tuning_trials, save_dir):
    diamond_df = pd.read_csv(dataset_path)
    diamond_df = basic_preprocess_diamonds(diamond_df)

    if model == "linear":
        train_dataset, test_dataset = preprocess_diamond_linear_reg(diamond_df)
        train_ev_model(train_dataset, test_dataset, LinearRegression, model, save_dir)

    elif model == "xgboost":
        train_dataset, test_dataset = preprocess_diamond_xgboost(diamond_df)
        model_kwargs = {
            "random_state": seed,
            "enable_categorical": True
        }
        if hyperparameter_tuning:
            model_kwargs.update(tune_hyperparameters(train_dataset, seed, n_tuning_trials))

        train_ev_model(train_dataset, test_dataset, xgboost.XGBRegressor, model, save_dir, **model_kwargs)


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
        default=DIR_PATH,
        help="specify different directory for models and reports",
    )

    args = parser.parse_args()
    print
    model = args.model
    dataset_path = args.dataset
    seed = args.seed
    tuning = args.tuning
    tuning_trials = args.tuning_trials
    save_dir = args.save_dir

    pipeline(model, dataset_path, tuning, seed, tuning_trials, save_dir)