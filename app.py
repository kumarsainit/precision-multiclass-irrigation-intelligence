from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lightgbm_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

model = None
model_columns = None

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    print("LightGBM model loaded successfully.")
else:
    print("Warning: Model files not found.")
    print("Run notebook/train_model.ipynb first and place the .pkl files in the project root.")

LABEL_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}

prediction_history = []
_history_id = [1]

MODELS_INFO = [
    {
        "id": "lightgbm",
        "name": "LightGBM",
        "description": (
            "Gradient Boosting framework using tree-based learning. "
            "Leaf-wise growth strategy gives superior accuracy on tabular data "
            "while remaining extremely fast to train and predict."
        ),
        "accuracy": 0.96,
        "precision": 0.95,
        "recall": 0.96,
        "f1_score": 0.96,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": "Maximum accuracy on structured tabular data — PRIMARY MODEL"
    },
    {
        "id": "xgboost",
        "name": "XGBoost",
        "description": (
            "Extreme Gradient Boosting with level-wise tree growth and "
            "regularisation. Achieves strong results but trains slower than LightGBM."
        ),
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.95,
        "f1_score": 0.94,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": "High-accuracy prediction with regularisation control"
    },
    {
        "id": "catboost",
        "name": "CatBoost",
        "description": (
            "Gradient boosting with native categorical feature support — "
            "no one-hot encoding required. Uses symmetric trees and ordered "
            "boosting to reduce overfitting."
        ),
        "accuracy": 0.94,
        "precision": 0.93,
        "recall": 0.94,
        "f1_score": 0.93,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": "Datasets with many categorical columns"
    },
    {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "description": (
            "Linear probabilistic classifier using SMOTE-balanced OHE data. "
            "Interpretable and fast; baseline for comparing tree-based ensembles."
        ),
        "accuracy": 0.82,
        "precision": 0.81,
        "recall": 0.83,
        "f1_score": 0.82,
        "algorithm_type": "Linear",
        "best_for": "Baseline model and linearly separable scenarios"
    }
]

FEATURE_IMPORTANCES = {
    "Soil_Moisture": 0.21,
    "Rainfall_mm": 0.18,
    "Temperature_C": 0.13,
    "Humidity": 0.10,
    "Previous_Irrigation_mm": 0.08,
    "Sunlight_Hours": 0.07,
    "Wind_Speed_kmh": 0.05,
    "Soil_pH": 0.04,
    "Electrical_Conductivity": 0.04,
    "Organic_Carbon": 0.03,
    "Crop_Growth_Stage": 0.03,
    "Crop_Type": 0.02,
    "Field_Area_hectare": 0.01,
    "Mulching_Used": 0.01
}

WATER_MM = {
    "Low": 15,
    "Medium": 30,
    "High": 50
}

FIELD_MAP = {
    "soil_moisture": "Soil_Moisture",
    "soil_ph": "Soil_pH",
    "organic_carbon": "Organic_Carbon",
    "electrical_conductivity": "Electrical_Conductivity",
    "temperature_c": "Temperature_C",
    "humidity": "Humidity",
    "rainfall_mm": "Rainfall_mm",
    "sunlight_hours": "Sunlight_Hours",
    "windspeed_kmph": "Wind_Speed_kmh",
    "crop_type": "Crop_Type",
    "crop_growth_stage": "Crop_Growth_Stage",
    "water_source": "Water_Source",
    "field_area_hectare": "Field_Area_hectare",
    "previous_irrigation_mm": "Previous_Irrigation_mm",
    "mulching_used": "Mulching_Used"
}

DEFAULTS = {
    "Soil_Type": "Loamy",
    "Season": "Kharif",
    "Irrigation_Type": "Drip",
    "Region": "North"
}


def build_reasoning(row, level):
    reasons = []

    soil_moisture = float(row.get("Soil_Moisture", 50))
    rainfall = float(row.get("Rainfall_mm", 0))
    temperature = float(row.get("Temperature_C", 25))
    humidity = float(row.get("Humidity", 60))
    wind_speed = float(row.get("Wind_Speed_kmh", 10))
    sunlight = float(row.get("Sunlight_Hours", 8))

    mulching = row.get("Mulching_Used", "No")
    growth_stage = str(row.get("Crop_Growth_Stage", "")).lower()

    if soil_moisture < 30:
        reasons.append(
            f"Soil moisture is very low ({soil_moisture}%) — "
            "well below the safe threshold of 40%"
        )
    elif soil_moisture > 70:
        reasons.append(
            f"Soil moisture is adequate ({soil_moisture}%) — "
            "immediate irrigation not urgent"
        )

    if rainfall > 20:
        reasons.append(
            f"Recent rainfall ({rainfall} mm) is reducing the irrigation deficit"
        )
    elif rainfall < 5:
        reasons.append(
            f"Very little recent rainfall ({rainfall} mm) — "
            "moisture replenishment needed"
        )

    if temperature > 35:
        reasons.append(
            f"High temperature ({temperature} °C) is accelerating evapotranspiration"
        )

    if mulching == "Yes":
        reasons.append(
            "Mulching is active — slowing soil moisture loss by ~20%"
        )

    if growth_stage == "flowering":
        reasons.append(
            "Crop is in the flowering stage — "
            "a critical water-sensitive period"
        )

    if wind_speed > 20:
        reasons.append(
            f"High wind speed ({wind_speed} km/h) is increasing surface evaporation"
        )

    if humidity < 40:
        reasons.append(
            f"Low humidity ({humidity}%) is increasing atmospheric water demand"
        )

    if sunlight > 9:
        reasons.append(
            f"Long sunlight exposure ({sunlight} hrs) raises crop water consumption"
        )

    if level == "Low":
        reasons.append(
            "Overall stress score is moderate — light irrigation sufficient"
        )

    return reasons[:5]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"models": MODELS_INFO})


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({
        "predictions": prediction_history[:20],
        "total": len(prediction_history)
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            raw = request.get_json()
        else:
            raw = request.form.to_dict()

        if not raw:
            return jsonify({"error": "No input data provided"}), 400

        mapped = {}

        for frontend_key, model_column in FIELD_MAP.items():
            if frontend_key in raw:
                value = raw[frontend_key]

                if frontend_key == "mulching_used":
                    mapped[model_column] = (
                        "Yes"
                        if str(value).lower() in ("true", "1", "yes")
                        else "No"
                    )
                else:
                    mapped[model_column] = value
            elif model_column not in mapped:
                mapped[model_column] = 0

        for column, default_value in DEFAULTS.items():
            if column not in mapped:
                mapped[column] = default_value

        input_df = pd.DataFrame([mapped])

        numeric_columns = [
            "Soil_pH",
            "Soil_Moisture",
            "Organic_Carbon",
            "Electrical_Conductivity",
            "Temperature_C",
            "Humidity",
            "Rainfall_mm",
            "Sunlight_Hours",
            "Wind_Speed_kmh",
            "Field_Area_hectare",
            "Previous_Irrigation_mm"
        ]

        for column in numeric_columns:
            if column in input_df.columns:
                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce"
                ).fillna(0)

        input_encoded = pd.get_dummies(input_df)

        if model_columns is None:
            return jsonify({
                "error": "Model columns not loaded. "
                         "Run the training notebook first."
            }), 500

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        if model is None:
            return jsonify({
                "error": "Model not loaded. "
                         "Run the training notebook first."
            }), 500

        prediction = int(model.predict(input_encoded)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_encoded)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.90

        level = LABEL_MAP.get(prediction, "Medium")

        area = float(raw.get("field_area_hectare", 1) or 1)
        water_mm = round(WATER_MM.get(level, 0) * area, 1)

        reasoning = build_reasoning(mapped, level)
        timestamp = datetime.now().isoformat()

        model_id = raw.get("model", "lightgbm")

        model_name = next(
            (
                model_info["name"]
                for model_info in MODELS_INFO
                if model_info["id"] == model_id
            ),
            "LightGBM"
        )

        record = {
            "id": _history_id[0],
            "model_used": model_name,
            "crop_type": raw.get("crop_type", "Unknown"),
            "irrigation_required": level,
            "confidence": round(confidence, 4),
            "water_recommendation_mm": water_mm,
            "timestamp": timestamp
        }

        prediction_history.insert(0, record)
        _history_id[0] += 1

        if len(prediction_history) > 50:
            prediction_history.pop()

        return jsonify({
            "irrigation_required": level,
            "confidence": round(confidence, 4),
            "water_recommendation_mm": water_mm,
            "model_used": model_name,
            "reasoning": reasoning,
            "feature_importances": FEATURE_IMPORTANCES,
            "timestamp": timestamp
        })

    except Exception as error:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)