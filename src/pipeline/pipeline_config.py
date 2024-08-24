import os
from dotenv import load_dotenv
import yaml

load_dotenv()
PIPELINE_CONFIG_DIR = os.getenv("PIPELINE_CONFIG_DIR")
MANDATORY_FIELDS = [
    "module",
    "class",
    "name",
]


class IncompleteConfigError(Exception):
    pass


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)

    for f in MANDATORY_FIELDS:
        if data.get(f, None) is None:
            raise IncompleteConfigError(
                f"Pipeline config file {config_path} is missing the {f} field."
            )

    return data
