import logging
import os
import pickle
from typing import Optional, Union
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from training_pipeline import DIR_PATH, basic_preprocess_diamonds

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)

# Change working directory so that code so that the behaviour
# is the same even if the code is executed from a different location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def preprocess_xgboost_sample(sample):
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


def preprocess_linear_sample(sample):
    global diamond_df
    df = diamond_df.copy().drop(columns="price")
    new_row = pd.DataFrame(sample, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop(columns=['depth', 'table', 'y', 'z'])
    df = pd.get_dummies(
        df, columns=['cut', 'color', 'clarity'], drop_first=True
    )
    return df.iloc[[-1]]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    logger.info("Loading dataset...")
    diamond_df = pd.read_csv(dataset_path)
    diamond_df = basic_preprocess_diamonds(diamond_df)
    logger.info("Dataset loaded.")
    return diamond_df


def load_best_model(
    directory: str,
    model_type: Optional[str] = None,
    criteria: str = "MAE"
) -> Union[LinearRegression, XGBRegressor]:

    try:
        report_df = pd.read_csv(os.path.join(directory, "report.csv"))

    except FileNotFoundError:
        raise RuntimeError((
            "No model report found. Make sure to train a model "
            "and put it in the right directory using training_pipeline.py"
        ))

    if model_type is not None:
        report_df = report_df.loc[report_df["type"] == model_type]

    if criteria == "MAE":
        ascending = True
    elif criteria == "r2":
        ascending = False

    ranking = report_df[criteria].sort_values(ascending=ascending).index
    best_model_id = report_df.loc[ranking[0], "model_id"]
    best_model_type = report_df.loc[ranking[0], "type"]

    with open(
        os.path.join(DIR_PATH, "model_files", best_model_id),
        "rb"
    ) as input_file:
        model = pickle.load(input_file)

    logger.info(
        "Loaded best %s model with %s of %f",
        best_model_type,
        criteria,
        report_df.loc[ranking[0], criteria]
    )

    return model, best_model_type
