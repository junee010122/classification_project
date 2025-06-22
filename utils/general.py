import os
import argparse
import yaml

from pydantic import ValidationError, BaseModel
from typing import Dict

class Params(BaseModel):
    paths: Dict[str, str]
    model: int
    hyperparameters: Dict[str, float | int]

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=None)
    return parser

def load_params():
    parser = parser_args()
    args, unknown = parser.parse_known_args()

    default_config = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "params.yaml"
    )

    config_file = args.configs if args.configs else default_config

    with open(config_file, "r") as f:
        raw_params = yaml.safe_load(f)

    try:
        params = Params(**raw_params)
    except ValidationError as e:
        print("Error: Invalid configuration parameters!")
        print(e)
        raise

    return params