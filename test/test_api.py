import requests
import numpy as np
from utils.general import load_params

def test_api():
    params = load_params()
    num_features = 590

    features = {f"feature_{i}": 0.0 for i in range(num_features)}

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=features)
        assert response.status_code == 200, f"API returned status code {response.status_code}"

        prediction = response.json().get("prediction")
        probability = response.json().get("probability")

        assert prediction is not None, "Prediction missing in API response"
        assert probability is not None, "Probability missing in API response"

        print(f"API Test Passed! Prediction: {prediction}, Probability: {probability}")

    except requests.exceptions.ConnectionError as e:
        print("Error: Could not connect to FastAPI server. Is it running?")
        print(e)


if __name__ == "__main__":
    test_api()
