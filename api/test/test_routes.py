"""
Test behavior of API routes
"""

from json import loads
from httpx import Response
from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_status() -> None:
    """
    Test API status endpoint
    """

    response: Response = client.get("/")

    assert response.status_code == 200, f"Error checking status: {response.status_code}"
    assert loads(response.content)["message"] == "API Running"


def test_predictions() -> None:
    """
    Test API ML model predictions
    """

    num_samples = 5
    num_features = 590
    data: list[dict[str, dict[str, float]]] = [
        {"data": {f"feature_{i}": 0.0 for i in range(num_features)}}
        for _ in range(num_samples)
    ]

    response: Response = client.post(
        "/predict", headers={"Content-Type": "application/json"}, json=data
    )

    assert response.status_code == 200, f"Error checking status: {response.status_code}"
