import os, json, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "crop_clean.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA)
needed = ["State_Name", "District_Name", "Crop", "Season", "Crop_Year", "Area", "Yield"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in crop_clean.csv: {missing}")

df = df[df["Crop"].isin(["Rice", "Maize"])].copy()
for c in ["Crop_Year", "Area", "Yield"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=needed)
df = df[(df["Area"] > 0) & (df["Yield"] >= 0) & (df["Yield"] <= 15)].copy()
if df.empty:
    raise ValueError("No usable Rice/Maize rows after cleaning.")

features = ["State_Name", "District_Name", "Crop", "Season", "Crop_Year", "Area"]
categorical = ["State_Name", "District_Name", "Crop", "Season"]
numeric = ["Crop_Year", "Area"]

def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])

def model_factories():
    return {
        "gradient_boosting": lambda: GradientBoostingRegressor(
            random_state=42, n_estimators=100, max_depth=3, learning_rate=0.05, loss="huber"
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42, n_jobs=2
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=2,
            random_state=42, n_jobs=2
        ),
    }

train = df[df["Crop_Year"] <= 2019]
test = df[df["Crop_Year"].between(2020, 2022)]
if train.empty or test.empty:
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

results = []
for name, factory in model_factories().items():
    validation_pipe = Pipeline([("pre", preprocessor()), ("model", factory())])
    validation_pipe.fit(train[features], train["Yield"])
    predicted = validation_pipe.predict(test[features])
    results.append({
        "model": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "MAE": mean_absolute_error(test["Yield"], predicted),
        "RMSE": mean_squared_error(test["Yield"], predicted) ** 0.5,
        "R2": r2_score(test["Yield"], predicted),
    })

    # Refit on all available data and save using the same filenames app.py loads.
    final_pipe = Pipeline([("pre", preprocessor()), ("model", factory())])
    final_pipe.fit(df[features], df["Yield"])
    joblib.dump(final_pipe, os.path.join(MODELS_DIR, name + ".joblib"))
    print(f"Saved models/{name}.joblib")

states = sorted(df["State_Name"].astype(str).unique().tolist())
districts = {
    s: sorted(df.loc[df["State_Name"].astype(str) == s, "District_Name"].astype(str).unique().tolist())
    for s in states
}
seasons = sorted(df["Season"].astype(str).unique().tolist())
with open(os.path.join(ROOT, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump({"states": states, "districts_by_state": districts, "seasons": seasons, "crops": ["Rice", "Maize"]}, f, indent=2)
pd.DataFrame(results).to_csv(os.path.join(ROOT, "model_comparison.csv"), index=False)
df.to_csv(os.path.join(ROOT, "rice_maize_training_data.csv"), index=False)

print("\nTraining complete. Validation results:")
print(pd.DataFrame(results).to_string(index=False))
print("\nNow run: python app.py")
