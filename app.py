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

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    print("XGBoost final deployment model loaded successfully.")

else:

    print("Warning: Final XGBoost model files not found.")
    print(
        "Run notebook/train_model.ipynb first "
        "and place xgboost_model.pkl and model_columns.pkl "
        "in the project root."
    )


LABEL_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}


prediction_history = []

_history_id = [1]


MODELS_INFO = [

    {
        "id": "xgboost",
        "name": "XGBoost",
        "description": (
            "Extreme Gradient Boosting classifier selected as the "
            "final deployment model after evaluating CatBoost, "
            "XGBoost, LightGBM, and Logistic Regression."
        ),
        "accuracy": 0.9970,
        "precision": 0.9970,
        "recall": 0.9970,
        "f1_score": 0.9970,
        "macro_roc_auc": 0.9976,
        "micro_average_precision": 0.9970,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": (
            "Final multi-class irrigation-need prediction "
            "with high generalization performance"
        ),
        "is_final": True
    },

    {
        "id": "catboost",
        "name": "CatBoost",
        "description": (
            "Gradient boosting classifier evaluated during model "
            "selection. It provided strong multi-class performance "
            "but did not exceed the final XGBoost evaluation."
        ),
        "accuracy": 0.9964,
        "precision": 0.9964,
        "recall": 0.9964,
        "f1_score": 0.9964,
        "macro_roc_auc": 0.9972,
        "micro_average_precision": 0.9964,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": (
            "Categorical-data modelling and comparative evaluation"
        ),
        "is_final": False
    },

    {
        "id": "lightgbm",
        "name": "LightGBM",
        "description": (
            "Gradient boosting classifier evaluated as part of "
            "the model comparison pipeline. It achieved performance "
            "very close to XGBoost but was not selected as the final model."
        ),
        "accuracy": 0.9969,
        "precision": 0.9969,
        "recall": 0.9969,
        "f1_score": 0.9969,
        "macro_roc_auc": 0.9975,
        "micro_average_precision": 0.9969,
        "algorithm_type": "Ensemble (Boosting)",
        "best_for": (
            "Fast gradient boosting on structured tabular data"
        ),
        "is_final": False
    },

    {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "description": (
            "Linear probabilistic classifier used as a baseline "
            "for comparison against tree-based ensemble models."
        ),
        "accuracy": 0.9034,
        "precision": 0.9034,
        "recall": 0.9034,
        "f1_score": 0.9034,
        "macro_roc_auc": 0.9297,
        "micro_average_precision": 0.9034,
        "algorithm_type": "Linear",
        "best_for": (
            "Baseline comparison and linearly separable scenarios"
        ),
        "is_final": False
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

    "electrical_conductivity":
        "Electrical_Conductivity",

    "temperature_c": "Temperature_C",

    "humidity": "Humidity",

    "rainfall_mm": "Rainfall_mm",

    "sunlight_hours": "Sunlight_Hours",

    "windspeed_kmph":
        "Wind_Speed_kmh",

    "crop_type": "Crop_Type",

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

    "Soil_Type": "Loamy",

    "Season": "Kharif",

    "Irrigation_Type": "Drip",

    "Region": "North"

}


def build_reasoning(row, level):

    reasons = []

    soil_moisture = float(
        row.get(
            "Soil_Moisture",
            50
        )
    )

    rainfall = float(
        row.get(
            "Rainfall_mm",
            0
        )
    )

    temperature = float(
        row.get(
            "Temperature_C",
            25
        )
    )

    humidity = float(
        row.get(
            "Humidity",
            60
        )
    )

    wind_speed = float(
        row.get(
            "Wind_Speed_kmh",
            10
        )
    )

    sunlight = float(
        row.get(
            "Sunlight_Hours",
            8
        )
    )

    mulching = row.get(
        "Mulching_Used",
        "No"
    )

    growth_stage = str(
        row.get(
            "Crop_Growth_Stage",
            ""
        )
    ).lower()


    if soil_moisture < 30:

        reasons.append(
            f"Soil moisture is very low "
            f"({soil_moisture}%) — well below "
            f"the safe threshold of 40%"
        )

    elif soil_moisture > 70:

        reasons.append(
            f"Soil moisture is adequate "
            f"({soil_moisture}%) — immediate "
            f"irrigation is not urgent"
        )


    if rainfall > 20:

        reasons.append(
            f"Recent rainfall ({rainfall} mm) "
            f"is reducing the irrigation deficit"
        )

    elif rainfall < 5:

        reasons.append(
            f"Very little recent rainfall "
            f"({rainfall} mm) — moisture "
            f"replenishment may be needed"
        )


    if temperature > 35:

        reasons.append(
            f"High temperature ({temperature} °C) "
            f"is accelerating evapotranspiration"
        )


    if mulching == "Yes":

        reasons.append(
            "Mulching is active — helping reduce "
            "soil moisture loss"
        )


    if growth_stage == "flowering":

        reasons.append(
            "Crop is in the flowering stage — "
            "a critical water-sensitive period"
        )


    if wind_speed > 20:

        reasons.append(
            f"High wind speed ({wind_speed} km/h) "
            f"is increasing surface evaporation"
        )


    if humidity < 40:

        reasons.append(
            f"Low humidity ({humidity}%) is "
            f"increasing atmospheric water demand"
        )


    if sunlight > 9:

        reasons.append(
            f"Long sunlight exposure ({sunlight} hrs) "
            f"raises crop water consumption"
        )


    if level == "Low":

        reasons.append(
            "Overall irrigation demand is low — "
            "light irrigation is sufficient"
        )


    if level == "Medium":

        reasons.append(
            "Overall irrigation demand is moderate — "
            "controlled irrigation is recommended"
        )


    if level == "High":

        reasons.append(
            "Overall irrigation demand is high — "
            "timely irrigation is recommended"
        )


    return reasons[:5]


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
        "models": MODELS_INFO,
        "final_model": "xgboost"
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
                    "No input data provided"
            }), 400


        if model is None:

            return jsonify({
                "error":
                    "Final XGBoost model is not loaded. "
                    "Run the training notebook first."
            }), 500


        if model_columns is None:

            return jsonify({
                "error":
                    "Model feature columns are not loaded. "
                    "Run the training notebook first."
            }), 500


        mapped = {}


        for frontend_key, model_column in FIELD_MAP.items():

            if frontend_key in raw:

                value = raw[frontend_key]


                if frontend_key == "mulching_used":

                    mapped[model_column] = (

                        "Yes"

                        if str(value).lower()
                        in (
                            "true",
                            "1",
                            "yes"
                        )

                        else "No"
                    )

                else:

                    mapped[model_column] = value


            elif model_column not in mapped:

                mapped[model_column] = 0


        for column, default_value in DEFAULTS.items():

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

        else:

            confidence = 0.0


        level = LABEL_MAP.get(
            prediction,
            "Medium"
        )


        area = float(
            raw.get(
                "field_area_hectare",
                1
            ) or 1
        )


        water_mm = round(

            WATER_MM.get(
                level,
                0
            ) * area,

            1

        )


        reasoning = build_reasoning(
            mapped,
            level
        )


        timestamp = datetime.now().isoformat()


        model_name = "XGBoost"


        record = {

            "id":
                _history_id[0],

            "model_used":
                model_name,

            "crop_type":
                raw.get(
                    "crop_type",
                    "Unknown"
                ),

            "irrigation_required":
                level,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "water_recommendation_mm":
                water_mm,

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


        return jsonify({

            "irrigation_required":
                level,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "water_recommendation_mm":
                water_mm,

            "model_used":
                model_name,

            "model_id":
                "xgboost",

            "reasoning":
                reasoning,

            "feature_importances":
                FEATURE_IMPORTANCES,

            "timestamp":
                timestamp

        })


    except Exception as error:

        import traceback

        traceback.print_exc()


        return jsonify({

            "error":
                "Prediction failed",

            "details":
                str(error)

        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
