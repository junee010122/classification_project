from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULT_PATH = "/Users/june/Documents/results/classification"

MODEL_PATH = os.path.join(RESULT_PATH, "best_model.pkl")
METRICS_PATH = os.path.join(RESULT_PATH, "model_metrics.csv")
PREDICTIONS_PATH = os.path.join(RESULT_PATH, "model_predictions.csv")

@app.get("/")
def read_root():
    return {"message": "Backend Running"}

@app.get("/metrics")
def get_metrics():
    try:
        df = pd.read_csv(METRICS_PATH)
        return df.to_dict(orient="records")[0]
    except Exception as e:
        return {"error": str(e)}

@app.get("/predictions")
def get_predictions():
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict(new_data: dict):
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.DataFrame([new_data])
        prob = model.predict_proba(df)[0][1]
        pred = int(prob >= 0.5)
        return {"prediction": pred, "probability": prob}
    except Exception as e:
        return {"error": str(e)}
