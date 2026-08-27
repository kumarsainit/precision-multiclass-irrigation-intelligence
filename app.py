from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgboost_model.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "model_columns.pkl"
)

model = None
model_columns = None


if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)

        print("XGBoost final deployment model loaded successfully.")
        print(f"Model feature columns loaded: {len(model_columns)}")

    except Exception as error:
        print("Error loading XGBoost model artifacts:")
        print(error)

else:
    print("Warning: Final XGBoost model artifacts were not found.")
    print("Required files:")
    print(" - xgboost_model.pkl")
    print(" - model_columns.pkl")
    print(
        "Run the training notebook first and place both artifacts "
        "in the project root."
    )


LABEL_MAP = {
    0: "High",
    1: "Low",
    2: "Medium"
}


prediction_history = []

_history_id = [1]


MODELS_INFO = [

    {
        "id": "xgboost",
        "name": "XGBoost",
        "description": (
            "Extreme Gradient Boosting classifier selected as the "
            "final production model after evaluating CatBoost, "
            "XGBoost, LightGBM, and Logistic Regression."
        ),
        "accuracy": 0.9853,
        "macro_f1": 0.9704,
        "high_f1": 0.9397,
        "macro_roc_auc": 0.9976,
        "micro_average_precision": 0.9970,
        "classification_report": {
            "High": {
                "precision": 0.9521,
                "recall": 0.9276,
                "f1_score": 0.9397
            },
            "Low": {
                "precision": 0.9867,
                "recall": 0.9951,
                "f1_score": 0.9909
            },
            "Medium": {
                "precision": 0.9860,
                "recall": 0.9752,
                "f1_score": 0.9806
            }
        },
        "algorithm_type": "Ensemble (Gradient Boosting)",
        "best_for": (
            "Final multi-class irrigation-need prediction "
            "with strong generalization and probability-based "
            "ranking performance."
        ),
        "is_final": True
    },

    {
        "id": "catboost",
        "name": "CatBoost",
        "description": (
            "Gradient boosting classifier evaluated during "
            "model selection. It achieved strong performance "
            "but was not selected as the final production model."
        ),
        "accuracy": 0.9853,
        "macro_f1": 0.9700,
        "high_f1": 0.9384,
        "macro_roc_auc": 0.9972,
        "micro_average_precision": 0.9964,
        "algorithm_type": "Ensemble (Gradient Boosting)",
        "best_for": (
            "Comparative evaluation and modelling "
            "of structured agricultural data."
        ),
        "is_final": False
    },

    {
        "id": "lightgbm",
        "name": "LightGBM",
        "description": (
            "Gradient boosting classifier evaluated during "
            "model comparison. It was not selected as the "
            "final production model."
        ),
        "accuracy": 0.9852,
        "macro_f1": 0.9705,
        "high_f1": 0.9403,
        "macro_roc_auc": 0.9975,
        "micro_average_precision": 0.9969,
        "algorithm_type": "Ensemble (Gradient Boosting)",
        "best_for": (
            "Comparative evaluation of fast gradient boosting "
            "on structured tabular data."
        ),
        "is_final": False
    },

    {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "description": (
            "Linear probabilistic classifier included as the "
            "baseline model for comparison against tree-based "
            "ensemble models."
        ),
        "accuracy": 0.8196,
        "macro_f1": 0.6894,
        "high_f1": 0.4202,
        "macro_roc_auc": 0.9297,
        "micro_average_precision": 0.9034,
        "algorithm_type": "Linear",
        "best_for": (
            "Baseline comparison and linearly separable "
            "classification scenarios."
        ),
        "is_final": False
    }

]


def get_feature_importances():

    if model is None or model_columns is None:
        return {}

    if not hasattr(model, "feature_importances_"):
        return {}

    try:

        importances = model.feature_importances_
        feature_names = list(model_columns)

        if len(importances) != len(feature_names):
            return {}

        feature_importance = {
            str(feature_names[index]): float(importances[index])
            for index in range(len(feature_names))
        }

        feature_importance = {
            key: value
            for key, value in feature_importance.items()
            if value > 0
        }

        return dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    except Exception as error:

        print(
            "Could not read XGBoost feature importance:",
            error
        )

        return {}


def build_reasoning(row, level):

    reasons = []

    try:
        soil_moisture = float(
            row.get("Soil_Moisture", 50)
        )
    except (TypeError, ValueError):
        soil_moisture = 50

    try:
        rainfall = float(
            row.get("Rainfall_mm", 0)
        )
    except (TypeError, ValueError):
        rainfall = 0

    try:
        temperature = float(
            row.get("Temperature_C", 25)
        )
    except (TypeError, ValueError):
        temperature = 25

    try:
        humidity = float(
            row.get("Humidity", 60)
        )
    except (TypeError, ValueError):
        humidity = 60

    try:
        wind_speed = float(
            row.get("Wind_Speed_kmh", 10)
        )
    except (TypeError, ValueError):
        wind_speed = 10

    try:
        sunlight = float(
            row.get("Sunlight_Hours", 8)
        )
    except (TypeError, ValueError):
        sunlight = 8

    mulching = str(
        row.get("Mulching_Used", "No")
    )

    growth_stage = str(
        row.get("Crop_Growth_Stage", "")
    ).lower()

    if soil_moisture < 30:
        reasons.append(
            f"Soil moisture is very low ({soil_moisture}%)."
        )

    elif soil_moisture > 70:
        reasons.append(
            f"Soil moisture is relatively high ({soil_moisture}%)."
        )

    if rainfall > 20:
        reasons.append(
            f"Recent rainfall is relatively high ({rainfall} mm)."
        )

    elif rainfall < 5:
        reasons.append(
            f"Recent rainfall is low ({rainfall} mm)."
        )

    if temperature > 35:
        reasons.append(
            f"High temperature ({temperature} °C) can increase "
            "crop water demand."
        )

    if humidity < 40:
        reasons.append(
            f"Low humidity ({humidity}%) can increase "
            "atmospheric water demand."
        )

    if wind_speed > 20:
        reasons.append(
            f"High wind speed ({wind_speed} km/h) can increase "
            "evaporation."
        )

    if sunlight > 9:
        reasons.append(
            f"Long sunlight exposure ({sunlight} hrs) can increase "
            "crop water consumption."
        )

    if mulching.lower() == "yes":
        reasons.append(
            "Mulching is active and can help reduce soil moisture loss."
        )

    if growth_stage == "flowering":
        reasons.append(
            "The crop is in the flowering stage, "
            "a water-sensitive growth period."
        )

    if level == "High":
        reasons.append(
            "XGBoost predicts a High irrigation-need class."
        )

    elif level == "Medium":
        reasons.append(
            "XGBoost predicts a Medium irrigation-need class."
        )

    elif level == "Low":
        reasons.append(
            "XGBoost predicts a Low irrigation-need class."
        )

    return reasons[:5]


FIELD_MAP = {

    "soil_moisture":
        "Soil_Moisture",

    "soil_ph":
        "Soil_pH",

    "organic_carbon":
        "Organic_Carbon",

    "electrical_conductivity":
        "Electrical_Conductivity",

    "temperature_c":
        "Temperature_C",

    "humidity":
        "Humidity",

    "rainfall_mm":
        "Rainfall_mm",

    "sunlight_hours":
        "Sunlight_Hours",

    "windspeed_kmph":
        "Wind_Speed_kmh",

    "crop_type":
        "Crop_Type",

    "crop_growth_stage":
        "Crop_Growth_Stage",

    "water_source":
        "Water_Source",

    "field_area_hectare":
        "Field_Area_hectare",

    "previous_irrigation_mm":
        "Previous_Irrigation_mm",

    "mulching_used":
        "Mulching_Used"

}


DEFAULTS = {

    "Soil_Type":
        "Loamy",

    "Season":
        "Kharif",

    "Irrigation_Type":
        "Drip",

    "Region":
        "North"

}


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/models")
def models_page():

    return render_template(
        "models.html"
    )


@app.route("/history")
def history_page():

    return render_template(
        "history.html"
    )


@app.route(
    "/api/models",
    methods=["GET"]
)
def get_models():

    return jsonify({

        "models":
            MODELS_INFO,

        "final_model":
            "xgboost",

        "final_model_name":
            "XGBoost",

        "final_test_metrics": {

            "accuracy":
                0.9853,

            "macro_f1":
                0.9704,

            "high_f1":
                0.9397,

            "macro_roc_auc":
                0.9976,

            "micro_average_precision":
                0.9970

        }

    })


@app.route(
    "/api/history",
    methods=["GET"]
)
def get_history():

    return jsonify({

        "predictions":
            prediction_history[:20],

        "total":
            len(prediction_history)

    })


@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        if request.is_json:
            raw = request.get_json()
        else:
            raw = request.form.to_dict()

        if not raw:
            return jsonify({
                "error":
                    "No input data provided."
            }), 400

        if model is None:
            return jsonify({
                "error":
                    "Final XGBoost model is not loaded.",
                "details":
                    (
                        "Ensure xgboost_model.pkl and "
                        "model_columns.pkl exist in the "
                        "project root."
                    )
            }), 500

        if model_columns is None:
            return jsonify({
                "error":
                    "Model feature columns are not loaded."
            }), 500

        mapped = {}

        for (
            frontend_key,
            model_column
        ) in FIELD_MAP.items():

            if frontend_key in raw:

                value = raw[frontend_key]

                if frontend_key == "mulching_used":

                    mapped[model_column] = (
                        "Yes"
                        if str(value).strip().lower()
                        in (
                            "true",
                            "1",
                            "yes",
                            "on"
                        )
                        else "No"
                    )

                else:

                    mapped[model_column] = value

        for (
            column,
            default_value
        ) in DEFAULTS.items():

            if column not in mapped:
                mapped[column] = default_value

        input_df = pd.DataFrame(
            [mapped]
        )

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

        input_encoded = pd.get_dummies(
            input_df
        )

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = int(
            model.predict(
                input_encoded
            )[0]
        )

        probabilities = None
        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_encoded
            )[0]

            confidence = float(
                max(probabilities)
            )

        level = LABEL_MAP.get(
            prediction
        )

        if level is None:

            return jsonify({
                "error":
                    (
                        "Unknown model output class "
                        f"received from XGBoost: {prediction}"
                    )
            }), 500

        
        water_recommendation_mm = None

        reasoning = build_reasoning(
            mapped,
            level
        )

        feature_importances = (
            get_feature_importances()
        )

        timestamp = datetime.now().isoformat()

        model_name = "XGBoost"
        model_id = "xgboost"

        record = {

            "id":
                _history_id[0],

            "model_used":
                model_name,

            "model_id":
                model_id,

            "crop_type":
                raw.get(
                    "crop_type",
                    "Unknown"
                ),

            "irrigation_required":
                level,

            "confidence":
                (
                    round(
                        confidence,
                        4
                    )
                    if confidence is not None
                    else None
                ),

            "water_recommendation_mm":
                water_recommendation_mm,

            "timestamp":
                timestamp

        }

        prediction_history.insert(
            0,
            record
        )

        _history_id[0] += 1

        if len(
            prediction_history
        ) > 50:

            prediction_history.pop()

        probability_details = {}

        if probabilities is not None:

            classes = getattr(
                model,
                "classes_",
                range(
                    len(probabilities)
                )
            )

            for index, class_id in enumerate(classes):

                try:
                    class_id_int = int(
                        class_id
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                class_name = LABEL_MAP.get(
                    class_id_int,
                    str(class_id_int)
                )

                probability_details[
                    class_name
                ] = round(
                    float(
                        probabilities[index]
                    ),
                    4
                )

        return jsonify({

            "irrigation_required":
                level,

            "predicted_class":
                prediction,

            "confidence":
                (
                    round(
                        confidence,
                        4
                    )
                    if confidence is not None
                    else None
                ),

            "class_probabilities":
                probability_details,

            "water_recommendation_mm":
                water_recommendation_mm,

            "water_recommendation_available":
                False,

            "model_used":
                model_name,

            "model_id":
                model_id,

            "reasoning":
                reasoning,

            "feature_importances":
                feature_importances,

            "timestamp":
                timestamp

        })

    except Exception as error:

        import traceback

        traceback.print_exc()

        return jsonify({

            "error":
                "Prediction failed.",

            "details":
                str(error)

        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )