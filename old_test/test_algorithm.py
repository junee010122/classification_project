import joblib
import pandas as pd
import numpy as np
from utils.general import load_params

def test_algorithm():
    params = load_params()
    model_path = params['paths']['model_save']
    model = joblib.load(model_path)

    num_features = 590
    feature_names = [f"feature_{i}" for i in range(num_features)]

    test_features = pd.DataFrame(
        [np.zeros(num_features)],
        columns=feature_names
    )

    prediction = model.predict(test_features)

    assert len(prediction) == 1
    print(f"Algorithm Test Passed! Prediction: {prediction[0]}")
