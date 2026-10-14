
from flask import Flask, render_template, request
import json
import os
import joblib
import pandas as pd

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "metadata.json"), encoding="utf-8") as f:
    META = json.load(f)

# Load the dataset to get available districts
data = pd.read_csv(os.path.join(ROOT, "crop_clean.csv"))
data = data[data["Crop"].isin(["Rice", "Maize"])]

districts = sorted(data["District_Name"].dropna().unique().tolist())

MODELS = {
    "Gradient Boosting": joblib.load(
        os.path.join(ROOT, "models", "gradient_boosting.joblib")
    ),
    "XGBoost": joblib.load(
        os.path.join(ROOT, "models", "xgboost.joblib")
    ),
    "Random Forest": joblib.load(
        os.path.join(ROOT, "models", "random_forest.joblib")
    )
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    form = {
        "state": "",
        "district": "",
        "crop": "Rice",
        "season": "Kharif",
        "year": "2022",
        "area": "1"
    }

    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in form}

        try:
            state = form["state"]
            district = form["district"]
            crop = form["crop"]
            season = form["season"]
            year = int(form["year"])
            area = float(form["area"])

            if state not in META["states"]:
                raise ValueError("Please select a valid state.")

            if district not in districts:
                raise ValueError("Please select a valid district.")

            if crop not in META["crops"]:
                raise ValueError("Choose Rice or Maize.")

            if season not in META["seasons"]:
                raise ValueError("Please select a valid season.")

            if area <= 0:
                raise ValueError("Area must be greater than zero.")

            if year < 1997 or year > 2035:
                raise ValueError("Enter a year from 1997 to 2035.")

            row = pd.DataFrame([{
                "State_Name": state,
                "District_Name": district,
                "Crop": crop,
                "Season": season,
                "Crop_Year": year,
                "Area": area
            }])

            predictions = {
                name: max(0.0, float(model.predict(row)[0]))
                for name, model in MODELS.items()
            }

            best = "XGBoost"

            result = {
                "predictions": predictions,
                "selected": best,
                "yield": predictions[best],
                "production": predictions[best] * area,
                "area": area
            }

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        meta=META,
        districts=districts,
        form=form,
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

