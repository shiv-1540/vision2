from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import medfilt

app = FastAPI()

class DataPoint(BaseModel):
    date: str
    meantemp: float
    humidity: float
    wind_speed: float
    meanpressure: float

class AnomalyDetectionRequest(BaseModel):
    data: List[DataPoint]


def load_and_preprocess_data(data: List[Dict]):
    df = pd.DataFrame(data)
    return df


def hampel_filter(series, window_size=7, n_sigma=3):
    median_filtered = medfilt(series, kernel_size=window_size)
    std_dev = np.std(series - median_filtered)
    outliers = np.abs(series - median_filtered) > n_sigma * std_dev
    return outliers


def detect_anomalies(df):
    df["z_score"] = np.abs(zscore(df["meantemp"]))
    df["z_anomaly"] = df["z_score"] > 3

    window = 7
    df["rolling_mean"] = df["meantemp"].rolling(window).mean()
    df["rolling_std"] = df["meantemp"].rolling(window).std()
    df["ma_anomaly"] = (df["meantemp"] > df["rolling_mean"] + 2 * df["rolling_std"]) | \
                       (df["meantemp"] < df["rolling_mean"] - 2 * df["rolling_std"])

    Q1 = df["meantemp"].quantile(0.25)
    Q3 = df["meantemp"].quantile(0.75)
    IQR = Q3 - Q1
    df["iqr_anomaly"] = (df["meantemp"] < (Q1 - 1.5 * IQR)) | (df["meantemp"] > (Q3 + 1.5 * IQR))

    df["hampel_anomaly"] = hampel_filter(df["meantemp"])

    methods = ["z_anomaly", "ma_anomaly", "iqr_anomaly", "hampel_anomaly"]
    summary = {method: int(df[method].sum()) for method in methods}
    best_method = max(summary, key=summary.get)
    
    return df, summary, best_method


@app.post("/detect-anomalies")
async def detect_anomalies_api(request: AnomalyDetectionRequest):
    try:
        df = load_and_preprocess_data([item.dict() for item in request.data])
        df, summary, best_method = detect_anomalies(df)
        
        return {
            "anomalies_detected": summary,
            "best_performing_method": best_method,
            "anomaly_records": df[df[list(summary.keys())].any(axis=1)].reset_index().to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_api():
    return {"message": "API is working!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

