import pandas as pd
from sklearn.model_selection import train_test_split


def basic_preprocess_diamonds(diamond_df: pd.DataFrame) -> pd.DataFrame:
    return diamond_df[
        (diamond_df.x * diamond_df.y * diamond_df.z != 0)
        & (diamond_df.price > 0)
        ]


def preprocess_diamond_linear_reg(
    diamond_df: pd.DataFrame,
    test_size: float,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    diamond_df = diamond_df.drop(columns=["depth", "table", "y", "z"])
    diamond_df = pd.get_dummies(
        diamond_df, columns=["cut", "color", "clarity"], drop_first=True
    )

    x = diamond_df.drop(columns="price")
    y = diamond_df.price
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed)

    train_dataset = x_train, y_train
    test_dataset = x_test, y_test

    return train_dataset, test_dataset


def preprocess_diamond_xgboost(
    diamond_df: pd.DataFrame,
    test_size: float,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    diamond_df = diamond_df.copy()
    diamond_df["cut"] = pd.Categorical(
        diamond_df["cut"],
        categories=["Fair", "Good", "Very Good", "Ideal", "Premium"],
        ordered=True
    )
    diamond_df["color"] = pd.Categorical(
        diamond_df["color"],
        categories=["D", "E", "F", "G", "H", "I", "J"],
        ordered=True
    )
    diamond_df["clarity"] = pd.Categorical(
        diamond_df["clarity"],
        categories=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
        ordered=True
    )

    x = diamond_df.drop(columns="price")
    y = diamond_df.price
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed)

    train_dataset = x_train, y_train
    test_dataset = x_test, y_test

    return train_dataset, test_dataset
