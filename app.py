from fastapi import FastAPI, HTTPException
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import medfilt

app = FastAPI()


def load_and_preprocess_data(data: List[Dict]):
    df = pd.DataFrame(data)
    return df


def hampel_filter(series, window_size=7, n_sigma=3):
    median_filtered = medfilt(series, kernel_size=window_size)
    std_dev = np.std(series - median_filtered)
    outliers = np.abs(series - median_filtered) > n_sigma * std_dev
    return outliers


def detect_anomalies(df):
    df["z_score"] = np.abs(zscore(df["meanpressure"]))
    df["z_anomaly"] = df["z_score"] > 3

    Q1 = df["meanpressure"].quantile(0.25)
    Q3 = df["meanpressure"].quantile(0.75)
    IQR = Q3 - Q1
    df["iqr_anomaly"] = (df["meanpressure"] < (Q1 - 1.5 * IQR)) | (df["meanpressure"] > (Q3 + 1.5 * IQR))

    df["hampel_anomaly"] = hampel_filter(df["meanpressure"])

    methods = ["z_anomaly", "iqr_anomaly", "hampel_anomaly"]
    summary = {method: int(df[method].sum()) for method in methods}
    best_method = max(summary, key=summary.get)
    
    return df, summary, best_method


@app.post("/detect-anomalies")
async def detect_anomalies_api(data: List[Dict]):
    try:
        df = load_and_preprocess_data(data)
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

# Now, you can hit the endpoint with JSON data directly! 🚀
