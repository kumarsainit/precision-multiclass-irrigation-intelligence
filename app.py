import json
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "irrigation_model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "model_schema.json")
RESULTS_PATH = os.path.join(BASE_DIR, "experiments", "results.json")

MAX_HISTORY = 50
HISTORY_PAGE_SIZE = 20

model = None
schema = None
results = None


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


try:
    model = joblib.load(MODEL_PATH)
    schema = _load_json(SCHEMA_PATH)
    results = _load_json(RESULTS_PATH)
    print(f"Loaded final model: {schema['final_model']}")
    print(f"Raw feature schema: {len(schema['raw_feature_order'])} columns")
except Exception as error:
    print("Failed to load deployment artifacts:", error)
    print("Run 'python3 experiments/run_experiment.py' to regenerate:")
    print(" - irrigation_model.pkl")
    print(" - model_schema.json")
    print(" - experiments/results.json")


ALGORITHM_TYPES = {
    "XGBoost": "Ensemble (gradient-boosted trees)",
    "LightGBM": "Ensemble (gradient-boosted trees)",
    "CatBoost": "Ensemble (gradient-boosted trees)",
    "Logistic Regression": "Linear",
}

MODEL_ROLES = {
    "XGBoost": (
        "Level-wise gradient boosting with explicit L1 and L2 penalties. Splits the "
        "feature space on thresholds, so it captures the non-linear soil-moisture and "
        "temperature cut-offs that separate the three irrigation classes."
    ),
    "LightGBM": (
        "Leaf-wise gradient boosting with histogram binning. It grows whichever leaf "
        "reduces the loss most, which trains fastest of the three ensembles on a "
        "dataset of this size."
    ),
    "CatBoost": (
        "Ordered boosting with symmetric trees. Included because the dataset carries "
        "eight categorical columns; for a fair comparison it was given the same one-hot "
        "matrix as the others, and its native categorical mode was measured separately."
    ),
    "Logistic Regression": (
        "Linear baseline. Establishes how much of the signal is linearly separable and "
        "how much genuinely requires a non-linear model."
    ),
}


def _model_id(name):
    return name.lower().replace(" ", "_")


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def build_models_info():
    if results is None or schema is None:
        return [], {}

    final_name = schema["final_model"]
    candidates = results["candidates"]
    test_metrics = results["all_candidates_test"]
    cv = results["cross_validation"]["results"]

    info = []
    for name in sorted(candidates, key=lambda n: candidates[n]["validation"]["macro_f1"], reverse=True):
        test = test_metrics[name]
        info.append(
            {
                "id": _model_id(name),
                "name": name,
                "description": MODEL_ROLES.get(name, ""),
                "algorithm_type": ALGORITHM_TYPES.get(name, "Unknown"),
                "is_final": name == final_name,
                "accuracy": _round(test["accuracy"]),
                "macro_f1": _round(test["macro_f1"]),
                "weighted_f1": _round(test["weighted_f1"]),
                "macro_precision": _round(test["macro_precision"]),
                "macro_recall": _round(test["macro_recall"]),
                "high_f1": _round(test["per_class"]["High"]["f1"]),
                "high_recall": _round(test["per_class"]["High"]["recall"]),
                "macro_roc_auc": _round(test.get("macro_roc_auc")),
                "micro_average_precision": _round(test.get("micro_average_precision")),
                "validation_macro_f1": _round(candidates[name]["validation"]["macro_f1"]),
                "validation_accuracy": _round(candidates[name]["validation"]["accuracy"]),
                "cv_macro_f1_mean": _round(cv[name]["macro_f1_mean"]),
                "cv_macro_f1_std": _round(cv[name]["macro_f1_std"]),
                "train_minus_val_macro_f1": _round(candidates[name]["gaps"]["train_minus_val_macro_f1"]),
                "classification_report": {
                    label: {
                        "precision": _round(scores["precision"]),
                        "recall": _round(scores["recall"]),
                        "f1_score": _round(scores["f1"]),
                        "support": scores["support"],
                    }
                    for label, scores in test["per_class"].items()
                },
            }
        )

    final_test = results["final_test"]
    summary = {
        "accuracy": _round(final_test["accuracy"]),
        "macro_precision": _round(final_test["macro_precision"]),
        "macro_recall": _round(final_test["macro_recall"]),
        "macro_f1": _round(final_test["macro_f1"]),
        "weighted_f1": _round(final_test["weighted_f1"]),
        "high_f1": _round(final_test["per_class"]["High"]["f1"]),
        "high_recall": _round(final_test["per_class"]["High"]["recall"]),
        "macro_roc_auc": _round(final_test.get("macro_roc_auc")),
        "micro_average_precision": _round(final_test.get("micro_average_precision")),
    }
    return info, summary


STRATEGY_LABELS = {
    "none": "No resampling",
    "class_weight_balanced": "Balanced class weights",
    "random_oversampling": "Random oversampling",
    "random_undersampling": "Random undersampling",
    "smote_on_encoded": "SMOTE to full balance",
    "smote_original_partial": "Partial SMOTE",
    "smotenc_on_raw": "SMOTENC",
    "smotetomek_on_encoded": "SMOTE with Tomek-link cleaning",
}


def describe_imbalance_strategy():
    """Describe the chosen strategy from the row counts it actually produced."""
    strategy = results["imbalance_study"]["strategies"][schema["imbalance_strategy"]]
    before = results["split"]["class_counts"]["train"]
    after = strategy["train_class_counts_after"]

    changed = [
        f"{label} from {before[label]:,} to {after[label]:,}"
        for label in schema["class_labels"]
        if after.get(label) != before.get(label)
    ]
    unchanged = [label for label in schema["class_labels"] if after.get(label) == before.get(label)]

    if not changed:
        return "The training rows were used exactly as observed, with no resampling."

    sentence = "Synthetic minority rows were interpolated to raise " + ", ".join(changed)
    if unchanged:
        sentence += ", leaving " + " and ".join(unchanged) + " at its observed count"
    return sentence + "."


def describe_regularisation():
    """List the overfitting controls that the final model was actually configured with."""
    name = schema["final_model"]
    params = results["candidate_hyperparameters"].get(name, {})
    labels = {
        "learning_rate": "shrinkage learning rate of {}",
        "depth": "tree depth capped at {}",
        "max_depth": "tree depth capped at {}",
        "num_leaves": "at most {} leaves per tree",
        "l2_leaf_reg": "L2 leaf regularisation of {}",
        "reg_lambda": "L2 penalty of {}",
        "reg_alpha": "L1 penalty of {}",
        "min_child_weight": "minimum child weight of {}",
        "min_child_samples": "at least {} samples per leaf",
        "subsample": "row subsampling at {}",
        "colsample_bytree": "column subsampling at {}",
        "C": "inverse regularisation strength of {}",
    }
    controls = [
        text.format(params[key])
        for key, text in labels.items()
        if key in params
    ]
    if params.get("early_stopping_rounds"):
        controls.insert(
            0,
            "early stopping against an inner slice of the training data, which keeps the "
            "validation split free for model selection",
        )
    controls.append(
        f"{results['cross_validation']['folds']}-fold stratified cross-validation to measure "
        "fold-to-fold variance"
    )
    return controls


def build_experiment_context():
    if results is None or schema is None:
        return {}

    final_name = schema["final_model"]
    overfitting = dict(results["overfitting"])
    overfitting["controls_applied"] = describe_regularisation()

    native = results.get("catboost_native_categorical", {})
    return {
        "final_model": final_name,
        "final_model_id": _model_id(final_name),
        "artifact": "irrigation_model.pkl",
        "schema_artifact": "model_schema.json",
        "imbalance_strategy": STRATEGY_LABELS.get(schema["imbalance_strategy"], schema["imbalance_strategy"]),
        "imbalance_strategy_key": schema["imbalance_strategy"],
        "imbalance_strategy_note": describe_imbalance_strategy(),
        "random_seed": schema["random_seed"],
        "dataset_rows": results["dataset"]["rows"],
        "class_distribution": results["dataset"]["class_distribution"],
        "imbalance_ratio": round(results["dataset"]["imbalance_ratio_majority_to_minority"], 2),
        "split": results["split"],
        "encoded_feature_count": results["preprocessing"]["encoded_feature_count"],
        "final_test": results["final_test"],
        "class_labels": schema["class_labels"],
        "overfitting": overfitting,
        "selection_rule": results["selection"]["rule"],
        "selection_ranking": results["selection"]["ranking"],
        "selection_margin": results["selection"]["margin_over_runner_up"],
        "cross_validation": results["cross_validation"],
        "catboost_native_macro_f1": native.get("validation", {}).get("macro_f1"),
        "environment": results["environment"],
        "top_features": list(results.get("feature_importance", {}).items())[:12],
    }


def get_feature_importances():
    if results is None:
        return {}
    return {
        name: value
        for name, value in results.get("feature_importance", {}).items()
        if value > 0
    }


def build_reasoning(row, level):
    """Explain a prediction against the training distribution recorded in the schema."""
    if schema is None:
        return []

    ranges = schema["numeric_ranges"]
    reasons = []

    def band(column):
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            return None, None
        stats = ranges[column]
        if value <= stats["p25"]:
            return value, "low"
        if value >= stats["p75"]:
            return value, "high"
        return value, "typical"

    moisture, moisture_band = band("Soil_Moisture")
    if moisture_band == "low":
        reasons.append(
            f"Soil moisture of {moisture:g} percent sits in the driest quarter of the "
            "training data."
        )
    elif moisture_band == "high":
        reasons.append(
            f"Soil moisture of {moisture:g} percent sits in the wettest quarter of the "
            "training data."
        )

    temperature, temperature_band = band("Temperature_C")
    if temperature_band == "high":
        reasons.append(
            f"Temperature of {temperature:g} degrees Celsius is in the top quarter of the "
            "training range, which raises evapotranspiration."
        )

    rainfall, rainfall_band = band("Rainfall_mm")
    if rainfall_band == "low":
        reasons.append(
            f"Cumulative rainfall of {rainfall:g} mm is in the lowest quarter of the "
            "training range."
        )
    elif rainfall_band == "high":
        reasons.append(
            f"Cumulative rainfall of {rainfall:g} mm is in the highest quarter of the "
            "training range."
        )

    wind, wind_band = band("Wind_Speed_kmh")
    if wind_band == "high":
        reasons.append(
            f"Wind speed of {wind:g} km/h is in the top quarter of the training range, "
            "which increases surface evaporation."
        )

    if str(row.get("Mulching_Used", "")).lower() == "yes":
        reasons.append("Mulching is active, and mulched fields skew towards lower irrigation need.")

    stage = str(row.get("Crop_Growth_Stage", ""))
    if stage in ("Flowering", "Vegetative"):
        reasons.append(f"The {stage.lower()} stage is the most water-sensitive period for the crop.")
    elif stage in ("Sowing", "Harvest"):
        reasons.append(f"The {stage.lower()} stage carries the lowest water demand in this dataset.")

    reasons.append(f"{schema['final_model']} assigns this field to the {level} irrigation-need class.")
    return reasons[:6]


FIELD_ALIASES = {
    "soil_type": "Soil_Type",
    "soil_ph": "Soil_pH",
    "soil_moisture": "Soil_Moisture",
    "organic_carbon": "Organic_Carbon",
    "electrical_conductivity": "Electrical_Conductivity",
    "temperature_c": "Temperature_C",
    "humidity": "Humidity",
    "rainfall_mm": "Rainfall_mm",
    "sunlight_hours": "Sunlight_Hours",
    "windspeed_kmph": "Wind_Speed_kmh",
    "wind_speed_kmh": "Wind_Speed_kmh",
    "crop_type": "Crop_Type",
    "crop_growth_stage": "Crop_Growth_Stage",
    "season": "Season",
    "irrigation_type": "Irrigation_Type",
    "water_source": "Water_Source",
    "field_area_hectare": "Field_Area_hectare",
    "mulching_used": "Mulching_Used",
    "previous_irrigation_mm": "Previous_Irrigation_mm",
    "region": "Region",
}

BOOLEAN_TRUE = {"true", "1", "yes", "on"}

prediction_history = []
_history_id = [1]


def normalise_payload(raw):
    """Map an incoming payload onto the exact raw training columns."""
    mapped = {}
    for key, value in raw.items():
        column = FIELD_ALIASES.get(str(key).strip().lower(), key)
        if column in schema["raw_feature_order"]:
            mapped[column] = value

    if "Mulching_Used" in mapped:
        mapped["Mulching_Used"] = (
            "Yes" if str(mapped["Mulching_Used"]).strip().lower() in BOOLEAN_TRUE else "No"
        )

    missing = [c for c in schema["raw_feature_order"] if c not in mapped]

    invalid = {}
    for column in schema["categorical_features"]:
        if column in mapped:
            value = str(mapped[column]).strip()
            levels = schema["categorical_levels"][column]
            if value not in levels:
                invalid[column] = {"received": value, "allowed": levels}
            else:
                mapped[column] = value

    for column in schema["numeric_features"]:
        if column in mapped:
            try:
                mapped[column] = float(mapped[column])
            except (TypeError, ValueError):
                invalid[column] = {"received": mapped[column], "expected": "a number"}

    return mapped, missing, invalid


@app.route("/")
def index():
    return render_template("index.html", schema=schema, experiment=build_experiment_context())


@app.route("/models")
def models_page():
    models_info, summary = build_models_info()
    return render_template(
        "models.html",
        schema=schema,
        experiment=build_experiment_context(),
        models=models_info,
        final_test_metrics=summary,
    )


@app.route("/history")
def history_page():
    return render_template("history.html", schema=schema, experiment=build_experiment_context())


@app.route("/api/models", methods=["GET"])
def get_models():
    if results is None or schema is None:
        return jsonify({"error": "Experiment results are not loaded."}), 500

    models_info, summary = build_models_info()
    return jsonify(
        {
            "models": models_info,
            "final_model": _model_id(schema["final_model"]),
            "final_model_name": schema["final_model"],
            "final_test_metrics": summary,
            "experiment": build_experiment_context(),
        }
    )


@app.route("/api/schema", methods=["GET"])
def get_schema():
    if schema is None:
        return jsonify({"error": "Model schema is not loaded."}), 500
    return jsonify(schema)


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(
        {
            "predictions": prediction_history[:HISTORY_PAGE_SIZE],
            "total": len(prediction_history),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or schema is None:
        return jsonify(
            {
                "error": "The deployment model is not loaded.",
                "details": "Run 'python3 experiments/run_experiment.py' to regenerate the artifacts.",
            }
        ), 500

    raw = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not raw:
        return jsonify({"error": "No input data provided."}), 400

    try:
        mapped, missing, invalid = normalise_payload(raw)

        if missing:
            return jsonify(
                {
                    "error": "Missing required field values.",
                    "missing_fields": missing,
                    "expected_fields": schema["raw_feature_order"],
                }
            ), 400

        if invalid:
            return jsonify({"error": "Invalid field values.", "invalid_fields": invalid}), 400

        frame = pd.DataFrame([mapped])[schema["raw_feature_order"]]

        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        confidence = float(max(probabilities))
        level = schema["class_index_to_label"][str(prediction)]

        class_probabilities = {
            schema["class_index_to_label"][str(int(class_id))]: round(float(probabilities[index]), 4)
            for index, class_id in enumerate(model.classes_)
        }

        timestamp = datetime.now().isoformat()
        prediction_history.insert(
            0,
            {
                "id": _history_id[0],
                "model_used": schema["final_model"],
                "model_id": _model_id(schema["final_model"]),
                "crop_type": mapped.get("Crop_Type", "Unknown"),
                "irrigation_required": level,
                "confidence": round(confidence, 4),
                "water_recommendation_mm": None,
                "timestamp": timestamp,
            },
        )
        _history_id[0] += 1
        del prediction_history[MAX_HISTORY:]

        return jsonify(
            {
                "irrigation_required": level,
                "predicted_class": prediction,
                "confidence": round(confidence, 4),
                "class_probabilities": class_probabilities,
                "water_recommendation_mm": None,
                "water_recommendation_available": False,
                "model_used": schema["final_model"],
                "model_id": _model_id(schema["final_model"]),
                "reasoning": build_reasoning(mapped, level),
                "feature_importances": get_feature_importances(),
                "features_used": mapped,
                "timestamp": timestamp,
            }
        )

    except Exception as error:
        import traceback

        traceback.print_exc()
        return jsonify({"error": "Prediction failed.", "details": str(error)}), 500


@app.errorhandler(404)
def page_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found", "status": 404}), 404
    return render_template("404.html", schema=schema, experiment=build_experiment_context()), 404


@app.errorhandler(500)
def server_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return render_template("500.html", schema=schema, experiment=build_experiment_context()), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
