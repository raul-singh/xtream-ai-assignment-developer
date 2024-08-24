from typing import Any
import pandas as pd

from assignment.pipeline.pipeline_config import load_config
from assignment.preprocessing.train_preprocess import preprocess_data


def preprocess_sample(
    sample: dict[str, Any],
    df: pd.DataFrame,
    config_path: str
) -> pd.DataFrame:

    pipeline_config = load_config(config_path)
    new_row = pd.DataFrame(sample, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    df, _ = preprocess_data(df, pipeline_config)
    return df.iloc[[-1]]
