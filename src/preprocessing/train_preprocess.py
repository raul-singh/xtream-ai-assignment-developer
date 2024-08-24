import logging
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def basic_preprocess_diamonds(diamond_df: pd.DataFrame) -> pd.DataFrame:
    return diamond_df[
        (diamond_df.x * diamond_df.y * diamond_df.z != 0)
        & (diamond_df.price > 0)
        ]


def preprocess_data(
    df: pd.DataFrame,
    pipeline_config: dict
) -> tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    df = df.copy()
    pipeline = pipeline_config["pipeline"]

    to_drop = pipeline.get("drop", [])
    dummies = pipeline.get("dummies", None)
    categorical = pipeline.get("categorical", [])
    target_variable = pipeline["target"]

    df = df.drop(columns=to_drop)

    if dummies is not None:
        df = pd.get_dummies(
            df, columns=dummies["columns"], drop_first=dummies["drop_first"]
        )

    for c in categorical:
        var = c["variable"]
        df[var] = pd.Categorical(
            df[var],
            categories=c["categories"],
            ordered=c["ordered"]
        )

    x = df.drop(columns=target_variable)
    y = df[target_variable]

    return x, y


def get_train_test_dataset(
    x: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    test_split_ratio: float,
    seed: int
) -> tuple[
    tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]],
    tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]
]:

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_split_ratio, random_state=seed)

    train_dataset = x_train, y_train
    test_dataset = x_test, y_test

    return train_dataset, test_dataset
