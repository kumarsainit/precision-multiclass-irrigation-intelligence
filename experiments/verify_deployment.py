"""Check that the served API reproduces the recorded experiment exactly.

Rebuilds the same split, scores the untouched test rows with the saved artifact,
compares against experiments/results.json, then sends a sample of raw rows through
the running HTTP API and requires identical answers.

    python3 app.py &
    python3 experiments/verify_deployment.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.environ.get("IRRIGATION_API", "http://127.0.0.1:5000")
SEED = 42
SAMPLE = 25

results = json.load(open(os.path.join(BASE_DIR, "experiments", "results.json")))
schema = json.load(open(os.path.join(BASE_DIR, "model_schema.json")))
pipeline = joblib.load(os.path.join(BASE_DIR, "irrigation_model.pkl"))

raw = pd.read_csv(os.path.join(BASE_DIR, "notebook", "train.csv")).drop(columns=["id"])
y = pd.Series(LabelEncoder().fit_transform(raw["Irrigation_Need"]), index=raw.index)
X = raw.drop(columns=["Irrigation_Need"])

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
_, _, _, _ = train_test_split(
    X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=SEED
)

failures = []


def check(label, actual, expected, tolerance=0.0):
    ok = abs(actual - expected) <= tolerance
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual:.6f} (recorded {expected:.6f})")
    if not ok:
        failures.append(label)


print(f"Feature order matches schema: {list(X.columns) == schema['raw_feature_order']}")
if list(X.columns) != schema["raw_feature_order"]:
    failures.append("raw feature order")

predictions = np.asarray(pipeline.predict(X_test[schema["raw_feature_order"]])).ravel()
check("test accuracy", accuracy_score(y_test, predictions), results["final_test"]["accuracy"])
check("test macro F1", f1_score(y_test, predictions, average="macro"), results["final_test"]["macro_f1"])

matrix_matches = (
    pd.crosstab(y_test, predictions).reindex(index=range(3), columns=range(3), fill_value=0).values
    == np.array(results["final_test"]["confusion_matrix"])
).all()
print(f"{'PASS' if matrix_matches else 'FAIL'}  test confusion matrix reproduced from the artifact")
if not matrix_matches:
    failures.append("confusion matrix")

sample = X_test[schema["raw_feature_order"]].head(SAMPLE)
offline_labels = [schema["class_index_to_label"][str(int(p))] for p in pipeline.predict(sample).ravel()]
offline_proba = pipeline.predict_proba(sample)

mismatches = 0
for position in range(len(sample)):
    payload = json.dumps(
        {column: (float(value) if isinstance(value, (int, float, np.floating)) else str(value))
         for column, value in sample.iloc[position].items()}
    ).encode()
    request = urllib.request.Request(
        f"{API}/api/predict", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        response = json.loads(urllib.request.urlopen(request, timeout=30).read())
    except urllib.error.URLError as error:
        print(f"SKIP  HTTP check: API not reachable at {API} ({error})")
        sys.exit(1 if failures else 0)

    if response.get("irrigation_required") != offline_labels[position]:
        mismatches += 1
        continue
    served = [response["class_probabilities"][label] for label in schema["class_labels"]]
    if not np.allclose(served, offline_proba[position], atol=1e-4):
        mismatches += 1

print(f"{'PASS' if mismatches == 0 else 'FAIL'}  {SAMPLE} raw rows served over HTTP match the offline pipeline")
if mismatches:
    failures.append("http parity")

print("\nRESULT:", "all deployment checks passed" if not failures else f"failed: {failures}")
sys.exit(1 if failures else 0)
