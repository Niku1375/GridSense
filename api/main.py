from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_artifact
from src.preprocess import engineer_features

app = FastAPI(title="GridSense API", description="Predicts hourly energy demand across regions.")

try:
    model = load_artifact("models/best_model.pkl")
    encoder = load_artifact("artifacts/region_encoder.pkl")
except Exception as e:
    model = None
    encoder = None
    print(f"Warning: Artifacts not loaded. {e}")

class DemandRequest(BaseModel):
    datetime: str
    region: str

@app.get("/health")
def health_check():
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model/Encoder unavailable.")
    return {"status": "healthy"}

@app.post("/predict")
def predict_demand(request: DemandRequest):
    df = pd.DataFrame([{"Datetime": request.datetime, "Region": request.region}])
    
    try:
        df = engineer_features(df, is_training=False)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown region: {request.region}")
        
    features = ['hour', 'dayofweek', 'month', 'Region_Code']
    X_input = df[features]
    prediction = model.predict(X_input)[0]
    
    return {
        "timestamp": request.datetime,
        "region": request.region,
        "predicted_demand_MW": float(prediction)
    }