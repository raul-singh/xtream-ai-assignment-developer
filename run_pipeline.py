import argparse
import logging
import os

from dotenv import load_dotenv
from src import pipeline

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        help="specify path of the pipeline"
    )

    args = parser.parse_args()
    pipeline_path = args.pipeline

    load_dotenv()
    model_dir_path = os.getenv("MODEL_DIR_PATH")
    dataset_path = os.getenv("DATASET_PATH")

    pipeline(pipeline_path, dataset_path, model_dir_path)


if __name__ == "__main__":
    main()
