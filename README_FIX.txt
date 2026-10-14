FIX THE MODEL LOADING ERROR (Windows / VS Code)

1. Copy train_models.py into your project root:
   D:\vscode\crop_yield_prediction_project\
   (same folder as app.py and requirements.txt)

2. Ensure crop_clean.csv is in that same folder. If it is not, copy it there.

3. In VS Code terminal, run:
   .venv\Scripts\activate

4. Install/update libraries inside that environment:
   python -m pip install --upgrade pip
   python -m pip install flask pandas scikit-learn xgboost joblib

5. Train compatible models locally:
   python train_models.py

6. Wait until it says "Training complete". Then start:
   python app.py

The script creates/replaces models\gradient_boosting.joblib,
models\xgboost.joblib, and models\random_forest.joblib in your own environment.
It also rewrites metadata.json and model_comparison.csv.

If app.py reports a missing package, install it in the activated .venv with:
   python -m pip install PACKAGE_NAME

Do not try to load the old joblib model files after retraining; the script overwrites them.
