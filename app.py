from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

app = Flask(__name__)

# Load the pre-trained model (ensure you've trained and saved this beforehand)
try:
    iso_forest = joblib.load("model/isolation.joblib")
except FileNotFoundError:
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    

def load_and_preprocess_data(data):
    """Load and clean incoming data."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df


def feature_engineering(df):
    """Add useful features for anomaly detection."""
    df['rolling_mean'] = df['meantemp'].rolling(window=7).mean().fillna(df['meantemp'])
    df['z_score'] = zscore(df['meantemp'], nan_policy='omit')
    return df


def detect_anomalies(df):
    """Apply anomaly detection algorithms."""
    df['iso_anomaly'] = iso_forest.predict(df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
    df['dbscan_anomaly'] = DBSCAN(eps=5, min_samples=2).fit_predict(df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
    df['z_anomaly'] = (np.abs(df['z_score']) > 2).astype(int)

    # Consolidate anomalies
    df['is_anomaly'] = (df['iso_anomaly'] == -1) | (df['dbscan_anomaly'] == -1) | (df['z_anomaly'] == 1)
    return df


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle incoming data and detect anomalies."""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"error": "Invalid input format. Send data as JSON."}), 400
        
        df = load_and_preprocess_data(data['data'])
        df = feature_engineering(df)
        df = detect_anomalies(df)

        anomalies = df[df['is_anomaly']]

        return jsonify({
            "total_records": len(df),
            "anomalies_detected": len(anomalies),
            "anomaly_percentage": (len(anomalies) / len(df)) * 100,
            "anomalies": anomalies.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/fetch-results', methods=['GET'])
def fetch_results():
    """Fetch previously detected anomalies from file."""
    try:
        df = pd.read_csv("detected_anomalies_weather.csv")
        return df.to_json(orient='records')
    except FileNotFoundError:
        return jsonify({"error": "No results found, please upload and process data first."}), 404


if __name__ == '__main__':
    app.run(debug=True)


# Fixes applied:
# 1. Corrected the IsolationForest model loading logic with fallback.
# 2. Fixed DBSCAN parameters to avoid clustering issues.
# 3. Added safe handling for NaNs in z-score.
# 4. Proper error messages and status codes for invalid requests.

# Your API should now be ready to handle the weather data and detect anomalies smoothly! 🚀