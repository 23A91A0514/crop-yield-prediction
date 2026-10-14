# Crop Yield Prediction: Rice and Maize

## Included
- Flask web interface with a state dropdown, crop, season, year, and cultivated area
- Trained Gradient Boosting, XGBoost, and Random Forest models
- Filtered training CSV and model comparison report

## Dataset preparation
- Original cleaned dataset contained 49,079 Rice/Maize rows.
- Training dataset contains 49,062 rows from 1997–2022.
- 17 records were excluded due to non-positive area or yield outside 0–15 tonnes/hectare. This is a simple cleaning rule; review outliers before production use.
- Target: `Yield` in tonnes/hectare. Input features: state, crop, season, crop year, area.
- The supplied data has no weather columns; no pesticide data is used.

## Holdout test
Trained on years through 2019 and tested on 2020–2022. Lower MAE/RMSE is better. XGBoost had the lowest RMSE in this test run.

            Model  Train rows  Test rows      MAE     RMSE       R2
Gradient Boosting       42062       7000 0.774979 1.170749 0.384522
          XGBoost       42062       7000 0.641155 0.928062 0.613242
    Random Forest       42062       7000 0.639664 0.930532 0.611181

## Run on Windows
1. Extract this ZIP.
2. Open Command Prompt in the extracted folder.
3. Run `py -m venv .venv`
4. Run `.venv\Scripts\activate`
5. Run `pip install -r requirements.txt`
6. Run `python app.py`
7. Open `http://127.0.0.1:5000` in your browser.

## Limitations
The dataset is historical and district-level records are aggregated in the app's feature set. Future-year predictions are extrapolations. Predictions are estimates, not guarantees. Validate with newer data before practical use.
