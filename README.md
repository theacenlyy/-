# Onaizah Temperature Forecast — Django App (Task 6)

A small Django app that wraps the SARIMA time-series model from Task 5
(`Re1.ipynb`) so anyone can pick a date in a browser and get a forecast
of the mean daily temperature for Onaizah, Saudi Arabia, with a 95%
confidence range.

> **Why a date, and not humidity/wind speed fields?**
> The Task 5 model is a **univariate SARIMA** model — it was trained only
> on the historical *date → temperature* series, not on other weather
> features. Asking the user for humidity or wind speed would collect
> input the model can't actually use. Instead, the one input the model
> genuinely needs is *how far into the future to forecast*, so the form
> asks for a target date and the app converts that into a forecast
> horizon internally.

## Project layout

```
weather_predictor/
├── manage.py
├── requirements.txt
├── train_and_save_model.py     # reproduces Task 5's pipeline, saves the model
├── make_demo_model.py          # fast synthetic model, for local testing only
├── models/                     # sarima_temperature_model.pkl lives here
├── weather_project/            # Django project (settings, urls, wsgi)
└── predictor/                  # the app
    ├── model_utils.py          # loads the pickle once, runs forecasts
    ├── forms.py                # validates the requested date
    ├── views.py
    ├── templates/predictor/
    └── static/predictor/css/style.css
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train (or reuse) the model

You need `models/sarima_temperature_model.pkl` before predictions will work.

**Option A — real model (matches the Task 5 notebook, needs internet):**

```bash
python train_and_save_model.py
```

This fetches the same Open-Meteo history the notebook uses, reruns
`auto_arima` to pick the SARIMA order, refits on the full history, and
saves the fitted `SARIMAXResults` object plus metadata (last training
date, order, seasonal order) into one pickle via `joblib`.

The equivalent cells are also appended to the end of `Re1.ipynb` — run
those instead if you'd rather do it from the notebook you already have
open.

**Option B — fast synthetic demo model (no internet, ~5 seconds), for
UI/testing purposes only:**

```bash
python make_demo_model.py
```

## Run locally

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000/`.

## How a prediction is served

1. `predictor/apps.py` loads the pickle once when the Django process
   starts (`AppConfig.ready()`), so the model isn't re-read from disk on
   every request.
2. `ForecastDateForm` (in `forms.py`) validates the submitted date:
   required, must parse as a real date, and must fall between "the day
   after the model's training history ends" and "365 days after that"
   (configurable via `MAX_FORECAST_DAYS_AHEAD` in settings). Dates
   outside that window, missing dates, and malformed input all produce
   an inline error next to the field instead of a crash.
3. `model_utils.predict_temperature()` converts the date into a forecast
   horizon (`steps = target_date - last_training_date`), calls
   `results.get_forecast(steps=...)`, and returns the point forecast
   plus a 95% confidence interval.
4. The result panel renders the number, the confidence range, and a
   small horizontal gauge showing where the forecast sits within that
   range.

To sanity-check the app's number against the notebook directly:

```python
import joblib
bundle = joblib.load("models/sarima_temperature_model.pkl")
bundle["results"].get_forecast(steps=N).predicted_mean.iloc[-1]
```
should match the app's output for a date `N` days after
`bundle["last_date"]`.

## Testing

```bash
python make_demo_model.py   # gives the test suite a model to validate against
python manage.py test
```

Manually verified:
- Valid date within range → forecast + confidence range render correctly.
- Empty submission → "Please choose a date." inline error, no crash.
- Non-date text → browser's native date picker mostly prevents this, but
  the server-side `DateField` also rejects it if bypassed.
- Date before the training history ends / more than a year past it →
  inline range error explaining the valid window.
- Missing/corrupt model file → the form disables submission and shows a
  clear "Model not loaded" message instead of a 500 error.
- Desktop (1280px) and mobile (375px) widths — the two-column layout
  collapses to a single column below 760px.

## Deploying publicly (optional)

Both Render and Railway can run this with minimal config:

1. Set environment variables from `.env.example`:
   `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`, `DJANGO_ALLOWED_HOSTS`.
2. Build: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
3. Start: `gunicorn weather_project.wsgi`
4. Static files are served in production by **WhiteNoise** (already
   wired up in `settings.py` / `MIDDLEWARE`), so no separate static
   file server is required.
5. Either commit `models/sarima_temperature_model.pkl` if it's small
   (a `SARIMAX` results object usually is — tens to low hundreds of KB),
   or set `WEATHER_MODEL_PATH` to point at a file you fetch from
   external storage during the build step.

## Bonus ideas not implemented (yet)

- A chart of the last N days of training history alongside the forecast.
- A `?format=json` query param on the same view for a machine-readable
  response.
- Caching identical `(date)` requests (e.g. Django's `cache_page`)
  since the model is deterministic for a given date.
