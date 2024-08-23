import argparse
import logging
import os
from dotenv import load_dotenv
from assignment import pipeline

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(name)s %(levelname)s: %(message)s",
    level=logging.INFO
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
        "--tuning-trials",
        type=int,
        nargs='?',
        default=100,
        help="how many hyperparameter tuning trials to perform",
    )

    parser.add_argument(
        "--test-split",
        type=float,
        nargs='?',
        default=0.2,
        help="the test/training split ratio.",
    )

    # Loading variables from .env file
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    MODEL_DIR_PATH = os.getenv("MODEL_DIR_PATH")

    args = parser.parse_args()
    model = args.model
    dataset_path = DATASET_PATH
    seed = args.seed
    tuning = args.tuning
    tuning_trials = args.tuning_trials
    save_dir = MODEL_DIR_PATH
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
