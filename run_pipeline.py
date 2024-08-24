import argparse
import logging
import os

from dotenv import load_dotenv
from assignment import pipeline

load_dotenv()
MODEL_DIR_PATH = os.getenv("MODEL_DIR_PATH")

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
    pipeline(pipeline_path, MODEL_DIR_PATH)


if __name__ == "__main__":
    main()
