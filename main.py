#!/usr/bin/env python3
"""
MAIN CONTROLLER for AIoT Smart Grid System (Part 1)
---------------------------------------------------
This script will:

1. Load forecasting + anomaly detection models
2. Fetch latest energy data (from ESP32 / API / CSV)
3. Perform load prediction
4. Detect anomalies
5. Run grid orchestration logic
6. Print actions for frontend or hardware layer
"""

import time
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.forecasting_model import LoadForecaster
from src.anomaly_detection import AnomalyDetector
from src.orchestration_engine import GridOrchestrator


def get_live_data():
    """
    TEMPORARY FUNCTION
    In real deployment, this will fetch live readings from:
    - ESP32 via API
    - MQTT broker
    - Cloud DB
    """
    try:
        df = pd.read_csv("data/sample_energy_data.csv")
        return df
    except FileNotFoundError:
        print("❌ sample_energy_data.csv not found in /data/")
        return None


def main_loop():
    print("🔌 Starting AIoT Smart Grid Controller...\n")

    # Load ML components
    forecaster = LoadForecaster()
    anomaly_detector = AnomalyDetector()
    orchestrator = GridOrchestrator()

    while True:
        print("📡 Fetching live data...")
        df = get_live_data()

        if df is None:
            time.sleep(5)
            continue

        # 1. Preprocess
        df_clean = preprocess_data(df)

        # 2. Feature Engineering
        df_features = create_features(df_clean)

        # 3. Predict Load
        predicted_load = forecaster.predict(df_features)
        print(f"🔮 Predicted Load: {predicted_load}")

        # 4. Detect Anomalies
        anomaly_flag = anomaly_detector.detect(df_features)
        print(f"🚨 Anomaly Detected: {anomaly_flag}")

        # 5. Grid Control Decision
        decision = orchestrator.decide(predicted_load, anomaly_flag)
        print(f"⚡ Grid Decision: {decision}\n")

        # Wait before next cycle
        time.sleep(5)


if __name__ == "__main__":
    main_loop()
