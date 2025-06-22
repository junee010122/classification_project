"""
Model tools
"""

from os import getenv

from joblib import load  # type: ignore
from pandas import DataFrame

from api.types import PredictionInput, PredictionOutput

path_model: str | None = getenv("MODEL_PATH")

if path_model:
    model = load(path_model)
else:
    raise RuntimeError(f"Error: {path_model} does not exist")


def process_data(data: list[PredictionInput]) -> list[PredictionOutput]:
    """
    Process single prediction input
    """

    data_dict: list[dict[str, float]] = [sample.model_dump()["data"] for sample in data]
    df: DataFrame = DataFrame(data_dict)

    predictions = model.predict_proba(df)

    results: list[PredictionOutput] = []
    for pred in predictions:
        f_pass, f_fail = pred
        eligibility = 1 if f_pass > f_fail else 0
        results.append(PredictionOutput(eligibility=eligibility, probability=pred))

    return results
