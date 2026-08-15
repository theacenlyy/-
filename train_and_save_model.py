"""
train_and_save_model.py

Re-creates the Task 5 pipeline (fetch Onaizah daily-mean-temperature
history, fit a SARIMA model, refit it on the *full* history) and saves
the result as a pickle the Django app (predictor/model_utils.py) can load.

This mirrors the "Save the model" cells appended to the end of Re1.ipynb —
run either the notebook cells or this script; they produce the same file.

Usage:
    python train_and_save_model.py
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import joblib
import pandas as pd
import requests
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

LATITUDE = 26.0843
LONGITUDE = 43.9936
START_DATE = "2019-01-01"
# تم التعديل: تثبيت التاريخ لنهاية الأرشيف لتفادي خطأ HTTP 400
END_DATE = "2023-12-31"

OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "sarima_temperature_model.pkl"


def fetch_history() -> pd.DataFrame:
    print(f"Fetching daily mean temperature for Onaizah ({START_DATE} -> {END_DATE})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Riyadh",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["daily"]["time"]),
            "temperature_2m_mean": data["daily"]["temperature_2m_mean"],
        }
    )
    df.set_index("date", inplace=True)
    df = df.asfreq("D")
    df["temperature_2m_mean"] = df["temperature_2m_mean"].interpolate(method="time")
    print(f"Fetched {len(df)} days of history.")
    return df


def train(df: pd.DataFrame):
    print("Searching for the best SARIMA order (this can take a few minutes)...")
    auto_model = auto_arima(
        df["temperature_2m_mean"],
        seasonal=True,
        m=7,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    print(f"Selected order={order}, seasonal_order={seasonal_order}")

    print("Refitting on the full history...")
    model = SARIMAX(
        df["temperature_2m_mean"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results, order, seasonal_order


def main():
    df = fetch_history()
    results, order, seasonal_order = train(df)

    bundle = {
        "results": results,
        "last_date": df.index[-1],
        "order": order,
        "seasonal_order": seasonal_order,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"Saved model bundle to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()