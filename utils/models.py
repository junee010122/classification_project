import os
import numpy as np
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression


def get_model(params=None):
    return LogisticRegression(**params) if params else LogisticRegression(max_iter=1000, class_weight='balanced')

from sklearn.impute import SimpleImputer

def train_model(X, y, params):
    result_path = params['paths']['result']
    os.makedirs(result_path, exist_ok=True)

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    num_features = X.shape[1]
    feature_names = [f"feature_{i}" for i in range(num_features)]
    X = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = get_model()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(result_path, "model_metrics.csv"), index=False)

    model_save_path = params['paths']['model_save']
    joblib.dump(model, model_save_path)

    results_df = X_test.copy()
    results_df["actual"] = y_test
    results_df["predicted"] = predictions
    results_df["probability"] = y_proba
    results_df.to_csv(os.path.join(result_path, "model_predictions.csv"), index=False)

    return model, results_df, metrics_df
