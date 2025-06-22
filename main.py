import os
import sys
import pickle as pkl
import numpy as np  
import pandas as pd
import joblib 

from utils.general import load_params
from utils.data import load_data
from utils.models import train_model
from utils import models


def run_experiment(params):
    data_path = params.paths["data"]
    X_full, y = load_data(data_path)
    
    results = []
    if params.model == 0:  # Logistic Regression
        model, result, metric = train_model(X_full, y, params.dict())
    print(f"Experiment completed. Metrics: {metric}")


if __name__ == "__main__":
    
    params = load_params() 
    run_experiment(params)
