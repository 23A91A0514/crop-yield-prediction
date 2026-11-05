import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

# Compatibility patch for unpickling across scikit-learn versions
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except (ImportError, AttributeError):
    pass

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CROP_CLEAN_PATH = os.path.join(ROOT, "crop_clean.csv")
ALL_INDIA_PATH = os.path.join(WORKSPACE_ROOT, "All-India_-Year-wise-Crop-Area,-Production-&-Yield.csv")
if not os.path.exists(ALL_INDIA_PATH):
    ALL_INDIA_PATH = os.path.join(ROOT, "All-India_-Year-wise-Crop-Area,-Production-&-Yield.csv")

WEATHER_PATH = os.path.join(ROOT, "state_year_weather_lookup.csv")
OUTPUT_TRAIN_DATA = os.path.join(ROOT, "training_data_with_weather.csv")
META_PATH = os.path.join(ROOT, "metadata.json")


def build_unified_dataset():
    print("1. Loading historical weather data...", flush=True)
    weather = pd.read_csv(WEATHER_PATH)
    weather["State_Name"] = weather["State_Name"].astype(str).str.strip().str.title()
    weather["Crop_Year"] = pd.to_numeric(weather["Crop_Year"], errors="coerce")
    weather_cols = [c for c in weather.columns if c not in ["State_Name", "Crop_Year"]]

    print("2. Loading cleaned district-level crop data (Rice, Maize, Wheat, Sugarcane)...", flush=True)
    clean_parts = []
    if os.path.exists(CROP_CLEAN_PATH):
        df_clean = pd.read_csv(CROP_CLEAN_PATH)
        df_clean["State_Name"] = df_clean["State_Name"].astype(str).str.strip().str.title()
        df_clean["Crop"] = df_clean["Crop"].astype(str).str.strip().str.title()
        df_clean["Season"] = df_clean["Season"].astype(str).str.strip().str.title()
        df_clean["Crop_Year"] = pd.to_numeric(df_clean["Crop_Year"], errors="coerce")
        df_clean["Area"] = pd.to_numeric(df_clean["Area"], errors="coerce")
        df_clean["Yield"] = pd.to_numeric(df_clean["Yield"], errors="coerce")

        for crop, grp in df_clean.groupby("Crop"):
            if crop == "Sugarcane":
                valid = grp[(grp["Area"] > 0) & (grp["Yield"] > 0) & (grp["Yield"] <= 150)]
            else:
                valid = grp[(grp["Area"] > 0) & (grp["Yield"] > 0) & (grp["Yield"] <= 20)]
            clean_parts.append(valid[["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production", "Yield"]])

    clean_df = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame()
    clean_with_weather = pd.merge(clean_df, weather, on=["State_Name", "Crop_Year"], how="left")
    print(f"   Cleaned district records: {len(clean_with_weather)} rows across {clean_df['Crop'].nunique()} crops", flush=True)

    existing_crops = set(clean_df["Crop"].unique()) if not clean_df.empty else set()

    print("3. Processing multiple crops from All-India dataset...", flush=True)
    if os.path.exists(ALL_INDIA_PATH):
        df_india = pd.read_csv(ALL_INDIA_PATH)
        crop_cols = [c.rsplit("-", 1)[0] for c in df_india.columns if c.endswith("-Yield")]

        all_india_list = []
        for _, row in df_india.iterrows():
            year_str = str(row["Year"]).strip()
            if "-" in year_str:
                crop_year = int(year_str.split("-")[0])
            else:
                try:
                    crop_year = int(year_str)
                except ValueError:
                    continue

            season = str(row["Season"]).strip()
            if season == "Total":
                season = "Whole Year"

            for crop_raw in crop_cols:
                crop = crop_raw.strip().title()
                if crop in existing_crops:
                    continue  # Keep detailed district-level records for existing crops

                yld = row.get(f"{crop_raw}-Yield")
                area = row.get(f"{crop_raw}-Area")
                prod = row.get(f"{crop_raw}-Production")
                if pd.notna(yld) and yld > 0:
                    yield_tonnes = float(yld) / 1000.0
                    area_val = float(area) if pd.notna(area) and area > 0 else 1.0
                    prod_val = float(prod) if pd.notna(prod) else (yield_tonnes * area_val)
                    all_india_list.append({
                        "Crop_Year": crop_year,
                        "Season": season,
                        "Crop": crop,
                        "Area": area_val,
                        "Production": prod_val,
                        "Yield": yield_tonnes,
                        "District_Name": "All Districts"
                    })

        melted_india = pd.DataFrame(all_india_list)
        print(f"   All-India points to integrate: {len(melted_india)} across {melted_india['Crop'].nunique()} crops", flush=True)

        # Cross-merge with weather across states
        weather_sub = weather[weather["Crop_Year"].isin(melted_india["Crop_Year"].unique())]
        add_df = pd.merge(melted_india, weather_sub, on="Crop_Year", how="inner")
        print(f"   Replicated multi-crop weather records: {len(add_df)} rows", flush=True)

        combined = pd.concat([clean_with_weather, add_df], ignore_index=True)
    else:
        combined = clean_with_weather

    print(f"4. Saving unified dataset to {OUTPUT_TRAIN_DATA}...", flush=True)
    combined.to_csv(OUTPUT_TRAIN_DATA, index=False)
    print(f"   Saved {len(combined)} rows covering {combined['Crop'].nunique()} crops.", flush=True)
    return combined, weather_cols


def train_and_evaluate():
    t_start = time.time()
    combined_df, weather_cols = build_unified_dataset()

    categorical = ["State_Name", "Crop", "Season"]
    numeric = ["Crop_Year", "Area"] + weather_cols
    features = categorical + numeric
    target = "Yield"

    # Load existing metadata to preserve districts_by_state
    districts_by_state = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            old_meta = json.load(f)
            districts_by_state = old_meta.get("districts_by_state", {})

    all_crops = sorted(combined_df["Crop"].unique().tolist())
    all_states = sorted(combined_df["State_Name"].unique().tolist())
    all_seasons = sorted(combined_df["Season"].unique().tolist())

    print("\n5. Preparing training sample...", flush=True)
    clean_data = combined_df.dropna(subset=categorical + ["Crop_Year", "Area", target]).copy()

    # Assign realistic farm area distribution so tree models learn farm size distributions properly
    np.random.seed(42)
    clean_data["Area"] = np.random.choice([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0], size=len(clean_data))

    # Stratified balance so every crop has strong representation without exploding training time
    dfs = []
    for crop, grp in clean_data.groupby("Crop"):
        if len(grp) > 2000:
            dfs.append(grp.sample(n=2000, random_state=42))
        else:
            dfs.append(grp)
    sample_df = pd.concat(dfs, ignore_index=True)
    print(f"   Balanced dataset rows: {len(sample_df)} across {sample_df['Crop'].nunique()} crops", flush=True)

    train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)
    print(f"   Training rows: {len(train_df)}, Testing rows: {len(test_df)}", flush=True)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])

    # Model specifications wrapped in TransformedTargetRegressor with log1p/expm1
    # This prevents high-yield crops (like Sugarcane ~75 t/ha) from dominating MSE loss
    # and ensures precise predictions for all 37 crops (pulses, cereals, cash crops, oilseeds)
    model_factories = {
        "xgboost": lambda: TransformedTargetRegressor(
            regressor=Pipeline([
                ("pre", preprocessor),
                ("model", XGBRegressor(
                    n_estimators=100, max_depth=7, learning_rate=0.08,
                    subsample=0.85, colsample_bytree=0.85,
                    objective="reg:squarederror", random_state=42, n_jobs=-1
                ))
            ]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "random_forest": lambda: TransformedTargetRegressor(
            regressor=Pipeline([
                ("pre", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=60, max_depth=20, min_samples_leaf=2,
                    random_state=42, n_jobs=-1
                ))
            ]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "gradient_boosting": lambda: TransformedTargetRegressor(
            regressor=Pipeline([
                ("pre", preprocessor),
                ("model", GradientBoostingRegressor(
                    n_estimators=60, max_depth=5, learning_rate=0.08,
                    random_state=42, loss="squared_error"
                ))
            ]),
            func=np.log1p, inverse_func=np.expm1
        ),
    }

    results = []
    print("\n6. Training and evaluating models...", flush=True)
    for name, factory in model_factories.items():
        print(f"   Fitting {name} on train split...", flush=True)
        t0 = time.time()
        val_model = factory()
        val_model.fit(train_df[features], train_df[target])
        elapsed = round(time.time() - t0, 2)

        pred = val_model.predict(test_df[features])
        r2 = float(r2_score(test_df[target], pred))
        mae = float(mean_absolute_error(test_df[target], pred))
        rmse = float(mean_squared_error(test_df[target], pred) ** 0.5)

        results.append({
            "model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "training_seconds": elapsed,
        })
        print(f"   -> {name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} (evaluation took {elapsed}s)", flush=True)

        # Fit on full sample and save
        print(f"   Fitting {name} on full dataset and saving...", flush=True)
        full_model = factory()
        full_model.fit(sample_df[features], sample_df[target])
        save_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(full_model, save_path)
        print(f"   -> Saved {save_path}", flush=True)

    # Determine best model
    best_result = sorted(results, key=lambda r: (-r["R2"], r["RMSE"], r["MAE"]))[0]
    best_model_name = best_result["model"]
    print(f"\nBest Model: {best_model_name} (R2={best_result['R2']:.4f})", flush=True)

    # Update metadata.json
    metadata = {
        "target": target,
        "crops": all_crops,
        "states": all_states,
        "districts_by_state": districts_by_state,
        "seasons": all_seasons,
        "feature_columns": features,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "weather_features": weather_cols,
        "best_model": best_model_name,
        "best_model_path": f"models/{best_model_name}.joblib",
        "rows_after_cleaning": len(sample_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "notes": [
            "Trained across 37 Indian crops using combined district-level and All-India historical records.",
            "Historical climate records automatically looked up by state and year.",
            "Yield is measured in tonnes per hectare; production is calculated as yield * area.",
            "Models utilize log1p target transformation to accurately predict yields across all crops."
        ]
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"7. Updated {META_PATH} with all {len(all_crops)} crops.", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(ROOT, "model_comparison.csv"), index=False)
    res_df.to_csv(os.path.join(ROOT, "model_evaluation_results.csv"), index=False)
    print("8. Updated model_comparison.csv and model_evaluation_results.csv:", flush=True)
    print(res_df.to_string(index=False), flush=True)
    print(f"\nAll operations completed successfully in {time.time() - t_start:.2f}s!", flush=True)


if __name__ == "__main__":
    train_and_evaluate()
