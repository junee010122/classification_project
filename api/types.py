"""
Typing tools
"""

from pydantic import BaseModel


class Status(BaseModel):
    """
    API status message
    """

    message: str


class PredictionOutput(BaseModel):
    """
    Model prediction attributes
    """

    eligibility: int
    probability: list[float]


class PredictionInput(BaseModel):
    """
    Model observation
    """

    data: dict[str, float]
