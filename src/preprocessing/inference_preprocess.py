from typing import Any
import pandas as pd
import numpy as np

from src.pipeline.pipeline_config import load_config
from src.preprocessing.train_preprocess import preprocess_data


def preprocess_sample(
    sample: dict[str, Any],
    df: pd.DataFrame,
    config_path: str
) -> pd.DataFrame:

    pipeline_config = load_config(config_path)
    sample["price"] = np.nan
    new_row = pd.DataFrame(sample, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    df, _ = preprocess_data(df, pipeline_config)
    return df.iloc[[-1]]
