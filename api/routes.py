"""
API routing tools
"""

from fastapi import APIRouter, status

from api.services.logistic_regression import process_data
from api.types import PredictionInput, PredictionOutput, Status


router = APIRouter()


@router.get("/", response_model=Status, status_code=status.HTTP_200_OK)
def run_status() -> Status:
    """
    Check the status of the API
    """

    return Status(message="API Running")


@router.post(
    "/predict", response_model=list[PredictionOutput], status_code=status.HTTP_200_OK
)
def run_predict(request: list[PredictionInput]) -> list[PredictionOutput]:
    """
    Get predictions from ML model
    """

    predictions: list[PredictionOutput] = process_data(data=request)

    return predictions


# @router.get("/metrics")
# def get_metrics():
#     try:
#         df = pd.read_csv(METRICS_PATH)
#         return df.to_dict(orient="records")[0]
#     except Exception as e:
#         return {"error": str(e)}
#
#
# @router.get("/predictions")
# def get_predictions():
#     try:
#         df = pd.read_csv(PREDICTIONS_PATH)
#         return df.to_dict(orient="records")
#     except Exception as e:
#         return {"error": str(e)}
