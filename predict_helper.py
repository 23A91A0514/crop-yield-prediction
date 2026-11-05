import sys
import os
from pathlib import Path
import json
import joblib
import pandas as pd

# Compatibility patch for unpickling gradient boosting across sklearn versions
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except (ImportError, AttributeError):
    pass

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))

# Load best model
best_model_rel = META.get("best_model_path", "models/random_forest.joblib")
MODEL = joblib.load(ROOT / best_model_rel)

# Load weather lookup table
WEATHER_FILE = ROOT / "state_year_weather_lookup.csv"
if WEATHER_FILE.exists():
    WEATHER = pd.read_csv(WEATHER_FILE)
    WEATHER["State_Name"] = WEATHER["State_Name"].astype(str).str.strip().str.title()
    WEATHER["Crop_Year"] = pd.to_numeric(WEATHER["Crop_Year"], errors="coerce")
else:
    WEATHER = pd.DataFrame(columns=["State_Name", "Crop_Year"] + META.get("weather_features", []))


def get_weather_features(state, year):
    """
    Retrieve weather features for a given state and year from the lookup table.
    Returns (weather_dict, is_matched).
    """
    state_norm = str(state).strip().title()
    year_norm = int(year)
    match = WEATHER[
        (WEATHER["State_Name"] == state_norm) &
        (WEATHER["Crop_Year"] == year_norm)
    ]
    weather_dict = {}
    is_matched = not match.empty
    for col in META["weather_features"]:
        val = match.iloc[0][col] if is_matched and pd.notna(match.iloc[0][col]) else float("nan")
        weather_dict[col] = float(val)
    return weather_dict, is_matched


def build_feature_dataframe(state, crop, season, year, area):
    """
    Build the full feature DataFrame expected by the trained pipeline models,
    automatically looking up weather features.
    """
    state_norm = str(state).strip().title()
    crop_norm = str(crop).strip().title()
    season_norm = str(season).strip().title()
    year_norm = int(year)
    area_val = float(area)

    weather_dict, is_matched = get_weather_features(state_norm, year_norm)

    row = {
        "State_Name": state_norm,
        "Crop": crop_norm,
        "Season": season_norm,
        "Crop_Year": year_norm,
        "Area": area_val,
        **weather_dict
    }
    X = pd.DataFrame([row], columns=META["feature_columns"])
    return X, is_matched, weather_dict


def predict_yield(state, crop, season, year, area):
    """
    Predict crop yield (tonnes/ha) using the best trained model (Random Forest).
    Automatically fetches weather features based on state and year.
    """
    X, _, _ = build_feature_dataframe(state, crop, season, year, area)
    return float(MODEL.predict(X)[0])
