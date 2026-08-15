"""
make_demo_model.py

Generates a small SYNTHETIC model bundle so you can test the Django app
end-to-end in a few seconds, without waiting on `train_and_save_model.py`
(which calls the Open-Meteo API and runs auto_arima — several minutes).

This is for local development/testing only. Swap in the real model
produced by train_and_save_model.py (or the notebook's "save model"
cells) before treating any prediction as meaningful.

Usage:
    python make_demo_model.py
"""

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "sarima_temperature_model.pkl"


def make_synthetic_series(days: int = 730) -> pd.Series:
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="D")
    day_of_year = dates.dayofyear.values
    # Rough Onaizah-like annual cycle: hot summers (~40C), mild winters (~13C)
    seasonal = 26 + 14 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    weekly_wobble = 0.6 * np.sin(2 * np.pi * dates.dayofweek.values / 7)
    noise = np.random.default_rng(7).normal(0, 1.2, size=days)
    values = seasonal + weekly_wobble + noise
    return pd.Series(values, index=dates, name="temperature_2m_mean")


def main():
    series = make_synthetic_series()
    print("Fitting a small demo SARIMAX(1,0,1)(1,0,1,7) model on synthetic data...")
    results = SARIMAX(
        series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    bundle = {
        "results": results,
        "last_date": series.index[-1],
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 1, 7),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "demo": True,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"Saved DEMO model bundle to {OUTPUT_PATH}")
    print("Replace it with train_and_save_model.py's output for real predictions.")


if __name__ == "__main__":
    main()
