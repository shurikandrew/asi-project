import os
import pandas as pd
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .scheme import ContinueTrainResponse, ContinueTrainRequest, PredictRequest, PredictResponse, ModelsResponse

app = FastAPI()
n_lags = 5

feature_names = [
    "Close_lag1",
    "Close_lag2",
    "Close_lag3",
    "Close_lag4",
    "Close_lag5"
]

@app.post("/continue-train", response_model=ContinueTrainResponse)
def continue_train(req: ContinueTrainRequest):
    models_folder = "src/model/joblib"
    old_model_path = os.path.join(models_folder, f"{req.model_name}.joblib")
    new_model_path = os.path.join(models_folder, f"{req.new_model_name}.joblib")

    if not os.path.exists(old_model_path):
        raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' does not exist")

    if os.path.exists(new_model_path):
        raise HTTPException(status_code=400, detail=f"Model '{req.new_model_name}' already exists")

    model = joblib.load(old_model_path)

    closes = [row.close for row in req.train_input]
    X_new = []
    y_new = []

    for i in range(n_lags, len(closes)):
        X_new.append(closes[i-n_lags:i])
        y_new.append(closes[i])

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    model.n_estimators += 100
    model.fit(X_new, y_new)

    joblib.dump(model, new_model_path)

    results = model.predict(X_new)

    rmse = np.sqrt(mean_squared_error(y_new, results))
    mae = mean_absolute_error(y_new, results)

    return {
        "metrics": {
            "rmse": rmse,
            "mae": mae,
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_path = f"src/model/joblib/{req.model_name}.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' does not exist")

    model = joblib.load(model_path)
    values = req.input

    X_pred = pd.DataFrame([values], columns=feature_names)
    y_pred = model.predict(X_pred)[0]

    prediction = y_pred

    return {"prediction": prediction}


@app.get("/models", response_model=ModelsResponse)
def get_models():
    files = os.listdir("src/model/joblib")
    models = [f.replace(".joblib", "") for f in files]
    return {"models": models}