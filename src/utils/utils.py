import logging
import os

import pandas as pd
from dotenv import load_dotenv

from src.preprocessing.train_preprocess import basic_preprocess_diamonds

# Create and initialize logger
logger = logging.getLogger(__name__)


def load_dataset(dataset_path) -> pd.DataFrame:
    logger.info("Loading dataset...")
    df = pd.read_csv(dataset_path)
    df = basic_preprocess_diamonds(df)
    logger.info("Dataset loaded and pre-processed.")
    return df


def extract_args_kwargs(args_kwargs_list: list) -> tuple[list, dict]:
    args = [arg for arg in args_kwargs_list if not isinstance(arg, dict)]
    pre_kwargs = [arg for arg in args_kwargs_list if isinstance(arg, dict)]
    kwargs = {}
    for single_dict in pre_kwargs:
        kwargs.update(single_dict)

    return args, kwargs
