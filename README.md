# All-India Crop Yield Prediction (37 Crops) with Automatic Weather Lookup

This project provides an interactive Flask web application that predicts crop yields (in tonnes per hectare) and total estimated production for **37 major Indian crops** across Indian states and seasons. It integrates machine-learning models (XGBoost, Random Forest, and Gradient Boosting) with an **automatic historical weather lookup** system.

---

## Supported Crops (37 Crops)
The application supports comprehensive yield and production forecasting for all major Indian crop categories:
- **Cereals & Foodgrains**: Rice, Wheat, Maize, Bajra, Barley, Jowar, Ragi, Small Millets, Cereals, Nutri/Coarse Cereals, Shree Anna / Nutri Cereals, Total Food Grains
- **Pulses**: Gram, Tur (Arhar), Moong, Urad, Lentil, Other Pulses, Total Pulses
- **Commercial & Cash Crops**: Sugarcane, Cotton, Tobacco, Jute, Mesta, Jute & Mesta, Sannhemp
- **Oilseeds**: Groundnut, Soybean, Sunflower, Rapeseed & Mustard, Sesamum, Castorseed, Linseed, Nigerseed, Safflower, Guarseed, Total Oil Seeds

---

## Key Features

1. **Automatic Weather Lookup**:
   - Historical temperature and rainfall data (1969–2024) are automatically looked up from `state_year_weather_lookup.csv` using the user's selected state and crop year.
   - Users do not need to manually input complex weather variables.
   - For future or unrecorded years (extrapolations), missing weather values are automatically imputed by the trained model pipeline.

2. **Multi-Model Comparison & Automatic Selection**:
   - Evaluates three machine learning algorithms: **XGBoost**, **Random Forest**, and **Gradient Boosting**.
   - Utilizes `TransformedTargetRegressor` with log-target transformation to accurately balance high-yield crops (e.g. Sugarcane ~70–80 t/ha) alongside standard grains and low-yield pulses/cotton (~0.4–3.5 t/ha).
   - XGBoost serves as the primary best model with high precision across distinct crops ($R^2 \approx 0.916$, $\text{MAE} \approx 0.481$).
   - Displays comparative predictions from all three models simultaneously.

3. **Interactive Web Interface**:
   - Dynamic state-dependent district selection.
   - Clean, responsive dashboard displaying predicted yield (tonnes/ha), total estimated production (tonnes), cultivated area, model evaluation metrics, and weather lookup status.

---

## File Structure

```text
crop_yield_prediction_project/
├── app.py                                # Main Flask web application
├── predict_helper.py                     # Standalone Python inference helper
├── train_models.py                       # Model training and evaluation script
├── test_app.py                           # Automated test suite
├── requirements.txt                      # Python dependencies
├── metadata.json                         # Feature configuration, 37 crops, states, districts, and model info
├── model_comparison.csv                  # Model evaluation metrics (R2, MAE, RMSE)
├── model_evaluation_results.csv          # Evaluation results backup
├── state_year_weather_lookup.csv         # Aggregated historical state-year weather dataset
├── weather_features_by_district_year.csv # District-year weather dataset
├── training_data_with_weather.csv        # Joined training dataset with weather features (209k rows)
├── crop_clean.csv                        # Cleaned historical crop dataset
├── rice_maize_training_data.csv          # Legacy dataset
├── models/
│   ├── xgboost.joblib                    # Fitted XGBoost pipeline (Primary best model)
│   ├── random_forest.joblib              # Fitted Random Forest pipeline
│   └── gradient_boosting.joblib          # Fitted Gradient Boosting pipeline
└── templates/
    └── index.html                        # Web UI template with dynamic forms and dashboard
```

---

## How to Run the Application

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Set Up a Virtual Environment (Optional but Recommended)
In Command Prompt / PowerShell inside the project directory:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Flask App
```bash
python app.py
```

### 5. Open in Your Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## Standalone Inference via Python
You can also run batch predictions or embed the models directly into custom scripts via `predict_helper.py`:

```python
from predict_helper import predict_yield

# Example 1: Predict yield for Sugarcane in Uttar Pradesh (10.0 hectares)
yield_sugar = predict_yield(
    state="Uttar Pradesh",
    crop="Sugarcane",
    season="Whole Year",
    year=2020,
    area=10.0
)
print(f"Predicted Sugarcane Yield: {yield_sugar:.3f} tonnes/ha (Production: {yield_sugar * 10.0:.2f} tonnes)")

# Example 2: Predict yield for Cotton in Gujarat (5.0 hectares)
yield_cotton = predict_yield(
    state="Gujarat",
    crop="Cotton",
    season="Kharif",
    year=2020,
    area=5.0
)
print(f"Predicted Cotton Yield: {yield_cotton:.3f} tonnes/ha (Production: {yield_cotton * 5.0:.2f} tonnes)")
```
Weather features are automatically retrieved from the historical weather lookup table behind the scenes.
