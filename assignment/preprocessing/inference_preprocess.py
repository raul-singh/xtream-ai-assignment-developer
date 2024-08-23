from typing import Any
import pandas as pd


def preprocess_xgboost_sample(sample: dict[str, Any]) -> pd.DataFrame:
    new_row = pd.DataFrame(sample, index=[0])

    new_row['cut'] = pd.Categorical(
        new_row['cut'],
        categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
        ordered=True
    )
    new_row['color'] = pd.Categorical(
        new_row['color'],
        categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
        ordered=True
    )
    new_row['clarity'] = pd.Categorical(
        new_row['clarity'],
        categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
        ordered=True
    )
    return new_row


def preprocess_linear_sample(
    sample: dict[str, Any],
    diamond_df: pd.DataFrame
) -> pd.DataFrame:

    df = diamond_df.copy().drop(columns="price")
    new_row = pd.DataFrame(sample, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop(columns=['depth', 'table', 'y', 'z'])
    df = pd.get_dummies(
        df, columns=['cut', 'color', 'clarity'], drop_first=True
    )
    return df.iloc[[-1]]
