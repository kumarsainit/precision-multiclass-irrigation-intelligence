"""Authoritative, leakage-free experiment for Precision Multi-Class Irrigation Intelligence.

Run from the project root:

    python3 experiments/run_experiment.py

Writes experiments/results.json plus the deployment artifacts irrigation_model.pkl
and model_schema.json. Every number quoted in README.md and MAIN.md comes from
experiments/results.json.
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
NOISE_SEEDS = [42, 7, 2024, 13, 101]
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EARLY_STOP_SIZE = 0.15
N_ESTIMATORS_CAP = 800
EARLY_STOPPING_ROUNDS = 50
CV_FOLDS = 5
CV_SUBSAMPLE = 150_000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "notebook", "train.csv")
OUT_DIR = os.path.join(BASE_DIR, "experiments")
TARGET = "Irrigation_Need"

# Ordered from fewest to most assumptions imposed on the training data. Used as the
# parsimony tie-break when several strategies are statistically indistinguishable.
STRATEGIES = [
    "none",
    "class_weight_balanced",
    "random_oversampling",
    "random_undersampling",
    "smote_on_encoded",
    "smote_original_partial",
    "smotenc_on_raw",
    "smotetomek_on_encoded",
]

np.random.seed(SEED)
results = {}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate(y_true, y_pred, y_proba, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(classes))
        ).tolist(),
    }
    if y_proba is not None:
        onehot = np.eye(len(classes))[np.asarray(y_true)]
        metrics["macro_roc_auc"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
        metrics["macro_average_precision"] = float(
            average_precision_score(onehot, y_proba, average="macro")
        )
        metrics["micro_average_precision"] = float(
            average_precision_score(onehot, y_proba, average="micro")
        )
    return metrics


# --------------------------------------------------------------- 1. load + audit
log("Loading dataset")
raw = pd.read_csv(DATA_PATH)
id_column_dropped = "id" in raw.columns
if id_column_dropped:
    raw = raw.drop(columns=["id"])

y_raw = raw[TARGET]
X_raw = raw.drop(columns=[TARGET])

CATEGORICAL = X_raw.select_dtypes(include="object").columns.tolist()
NUMERIC = [c for c in X_raw.columns if c not in CATEGORICAL]

label_encoder = LabelEncoder().fit(y_raw)
CLASSES = label_encoder.classes_.tolist()
y_all = pd.Series(label_encoder.transform(y_raw), index=X_raw.index)
HIGH_INDEX = CLASSES.index("High")

class_counts = y_raw.value_counts()
duplicate_features = int(X_raw.duplicated().sum())
contradictory = 0
if duplicate_features:
    dup_mask = X_raw.duplicated(keep=False)
    grouped = raw[dup_mask].groupby(list(X_raw.columns), observed=True)[TARGET].nunique()
    contradictory = int((grouped > 1).sum())

results["dataset"] = {
    "path": "notebook/train.csv",
    "rows": int(len(raw)),
    "n_features": int(X_raw.shape[1]),
    "id_column_dropped": id_column_dropped,
    "target": TARGET,
    "classes": CLASSES,
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "categorical_levels": {c: sorted(X_raw[c].unique().tolist()) for c in CATEGORICAL},
    "numeric_ranges": {
        c: {
            "min": float(X_raw[c].min()),
            "p25": float(X_raw[c].quantile(0.25)),
            "median": float(X_raw[c].median()),
            "p75": float(X_raw[c].quantile(0.75)),
            "max": float(X_raw[c].max()),
            "mean": float(X_raw[c].mean()),
            "std": float(X_raw[c].std()),
        }
        for c in NUMERIC
    },
    "missing_values_total": int(raw.isnull().sum().sum()),
    "duplicate_rows_including_target": int(raw.duplicated().sum()),
    "duplicate_rows_features_only": duplicate_features,
    "contradictory_label_groups": contradictory,
    "class_distribution": {
        cls: {
            "count": int(class_counts[cls]),
            "percentage": float(class_counts[cls] / len(raw) * 100),
        }
        for cls in CLASSES
    },
    "imbalance_ratio_majority_to_minority": float(class_counts.max() / class_counts.min()),
}

# class-conditional means, used to explain which features carry the signal
results["dataset"]["numeric_mean_by_class"] = {
    cls: {c: float(v) for c, v in raw[y_raw == cls][NUMERIC].mean().items()} for cls in CLASSES
}
results["dataset"]["categorical_class_share"] = {
    c: (pd.crosstab(raw[c], y_raw, normalize="index") * 100).round(3).to_dict("index")
    for c in CATEGORICAL
}

log(
    f"rows={len(raw)} classes={CLASSES} "
    f"imbalance={results['dataset']['imbalance_ratio_majority_to_minority']:.2f}"
)

# --------------------------------------------------------------- 2. split
log("Splitting raw rows 70/15/15 (stratified) before any preprocessing")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=VAL_SIZE / (1.0 - TEST_SIZE),
    stratify=y_trainval,
    random_state=SEED,
)

overlap_train_val = len(set(X_train.index) & set(X_val.index))
overlap_train_test = len(set(X_train.index) & set(X_test.index))
overlap_val_test = len(set(X_val.index) & set(X_test.index))
assert overlap_train_val == overlap_train_test == overlap_val_test == 0

results["split"] = {
    "strategy": "stratified random split of raw rows, performed before any encoding or resampling",
    "random_seed": SEED,
    "train_rows": int(len(X_train)),
    "val_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "train_pct": round(len(X_train) / len(raw) * 100, 2),
    "val_pct": round(len(X_val) / len(raw) * 100, 2),
    "test_pct": round(len(X_test) / len(raw) * 100, 2),
    "index_overlap_train_val": overlap_train_val,
    "index_overlap_train_test": overlap_train_test,
    "index_overlap_val_test": overlap_val_test,
    "class_counts": {
        "train": {CLASSES[k]: int(v) for k, v in y_train.value_counts().items()},
        "validation": {CLASSES[k]: int(v) for k, v in y_val.value_counts().items()},
        "test": {CLASSES[k]: int(v) for k, v in y_test.value_counts().items()},
    },
}
log(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# --------------------------------------------- 3. preprocessing fitted on train only
def make_preprocessor(scale_numeric=False):
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC),
        ]
    )


tree_preprocessor = make_preprocessor().fit(X_train)
FEATURE_NAMES = [n.split("__", 1)[1] for n in tree_preprocessor.get_feature_names_out()]
Xt_train = tree_preprocessor.transform(X_train)
Xt_val = tree_preprocessor.transform(X_val)
Xt_test = tree_preprocessor.transform(X_test)

results["preprocessing"] = {
    "fitted_on": "training split only",
    "categorical_encoding": "OneHotEncoder(drop='first', handle_unknown='ignore')",
    "numeric_handling": (
        "passed through unscaled for the tree ensembles; StandardScaler applied inside the "
        "Logistic Regression pipeline only, because a linear model needs comparable scales to converge"
    ),
    "encoded_feature_count": int(Xt_train.shape[1]),
    "encoded_feature_names": FEATURE_NAMES,
}

# --------------------------------------------------------------- 4. imbalance study
PROBE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def apply_strategy(X_enc, y, strategy, seed, raw_frame=None, encoder=None):
    """Apply an imbalance strategy to an already-encoded training matrix.

    Returns (X, y, sample_weight). raw_frame/encoder are only needed by SMOTENC, which
    has to see the untouched categorical columns.
    """
    if strategy == "none":
        return X_enc, y, None

    if strategy == "class_weight_balanced":
        counts = np.bincount(y)
        return X_enc, y, (len(y) / (len(counts) * counts))[y]

    if strategy == "random_oversampling":
        Xr, yr = RandomOverSampler(random_state=seed).fit_resample(X_enc, y)
    elif strategy == "random_undersampling":
        Xr, yr = RandomUnderSampler(random_state=seed).fit_resample(X_enc, y)
    elif strategy == "smote_on_encoded":
        Xr, yr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X_enc, y)
    elif strategy == "smote_original_partial":
        counts = np.bincount(y)
        scale = len(y) / len(y_train)
        targets = {
            HIGH_INDEX: max(int(counts[HIGH_INDEX]), int(100_000 * scale)),
            CLASSES.index("Medium"): max(int(counts[CLASSES.index("Medium")]), int(200_000 * scale)),
        }
        Xr, yr = SMOTE(sampling_strategy=targets, random_state=seed, k_neighbors=5).fit_resample(X_enc, y)
    elif strategy == "smotenc_on_raw":
        Xr_df, yr = SMOTENC(
            categorical_features=CATEGORICAL, random_state=seed, k_neighbors=5
        ).fit_resample(raw_frame, y)
        Xr = encoder.transform(Xr_df)
    elif strategy == "smotetomek_on_encoded":
        Xr, yr = SMOTETomek(random_state=seed).fit_resample(X_enc, y)
    else:
        raise ValueError(strategy)
    return Xr, yr, None


def resample(strategy, seed=SEED):
    return apply_strategy(
        Xt_train, y_train.values, strategy, seed, raw_frame=X_train, encoder=tree_preprocessor
    )


STRATEGY_NOTES = {
    "none": "train on the observed distribution, no resampling and no reweighting",
    "class_weight_balanced": "no synthetic rows; each class weighted inversely to its frequency",
    "random_oversampling": "existing minority rows duplicated until every class matches the majority",
    "random_undersampling": "majority rows discarded until every class matches the minority",
    "smote_on_encoded": "plain SMOTE interpolating the one-hot columns, minority classes raised to majority size",
    "smote_original_partial": "the original project's partial SMOTE targets, also interpolating one-hot columns",
    "smotenc_on_raw": "SMOTENC on the raw frame: numeric columns interpolated, categorical columns set by neighbourhood majority vote",
    "smotetomek_on_encoded": "SMOTE followed by Tomek-link cleaning of borderline pairs",
}

log("Imbalance strategy study (probe model: LightGBM with fixed hyperparameters)")
strategy_results = {}
for strategy in STRATEGIES:
    start = time.time()
    try:
        Xr, yr, weights = resample(strategy)
        model = LGBMClassifier(random_state=SEED, **PROBE_PARAMS)
        model.fit(Xr, yr, sample_weight=weights)
        metrics = evaluate(y_val.values, model.predict(Xt_val), model.predict_proba(Xt_val), CLASSES)
    except Exception as error:  # noqa: BLE001
        log(f"  {strategy}: FAILED -> {error}")
        strategy_results[strategy] = {"status": "failed", "error": str(error)}
        continue

    strategy_results[strategy] = {
        "status": "ok",
        "note": STRATEGY_NOTES[strategy],
        "train_rows_after": int(len(yr)),
        "train_class_counts_after": {
            CLASSES[k]: int(v) for k, v in zip(*np.unique(yr, return_counts=True))
        },
        "validation": metrics,
        "seconds": round(time.time() - start, 1),
    }
    log(
        f"  {strategy:<24} rows={len(yr):>7} macroF1={metrics['macro_f1']:.4f} "
        f"acc={metrics['accuracy']:.4f} HighRecall={metrics['per_class']['High']['recall']:.4f} "
        f"HighF1={metrics['per_class']['High']['f1']:.4f}"
    )

ok = {k: v for k, v in strategy_results.items() if v.get("status") == "ok"}
top_macro = max(v["validation"]["macro_f1"] for v in ok.values())

# noise floor: repeat the two cheapest strategies across seeds to measure how much
# validation Macro F1 moves for reasons unrelated to the strategy itself
log("Run-to-run noise study")
noise = {}
for strategy in ("none", "smote_original_partial"):
    scores = []
    for seed in NOISE_SEEDS:
        Xr, yr, weights = resample(strategy, seed)
        model = LGBMClassifier(random_state=seed, **PROBE_PARAMS)
        model.fit(Xr, yr, sample_weight=weights)
        scores.append(float(f1_score(y_val.values, model.predict(Xt_val), average="macro")))
    noise[strategy] = {
        "seeds": NOISE_SEEDS,
        "macro_f1_scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "range": float(max(scores) - min(scores)),
    }
    log(f"  {strategy:<24} mean={np.mean(scores):.4f} std={np.std(scores, ddof=1):.4f}")

NOISE_STD = max(v["std"] for v in noise.values())
threshold = top_macro - NOISE_STD
indistinguishable = [
    s for s in STRATEGIES if s in ok and ok[s]["validation"]["macro_f1"] >= threshold
]
BEST_STRATEGY = indistinguishable[0]

results["imbalance_study"] = {
    "probe_model": "LightGBM with fixed hyperparameters, trained on the training split and scored on validation",
    "probe_hyperparameters": PROBE_PARAMS,
    "assumption_order": STRATEGIES,
    "selection_rule": (
        "Take the best validation Macro F1. Estimate the run-to-run standard deviation of that metric "
        "by repeating the probe across five seeds. Every strategy within one standard deviation of the "
        "best is treated as statistically indistinguishable, and among those the strategy that imposes "
        "the fewest assumptions on the training data is selected."
    ),
    "strategies": strategy_results,
    "noise_study": noise,
    "noise_std_used": NOISE_STD,
    "best_validation_macro_f1": top_macro,
    "indistinguishable_threshold": threshold,
    "indistinguishable_strategies": indistinguishable,
    "selected": BEST_STRATEGY,
}
log(f"Selected imbalance strategy: {BEST_STRATEGY}")

Xr_train, yr_train, w_train = resample(BEST_STRATEGY)

# --------------------------------------------------------------- 5. candidates
log("Training candidates under an identical protocol")
fit_idx, stop_idx = train_test_split(
    np.arange(len(yr_train)), test_size=EARLY_STOP_SIZE, stratify=yr_train, random_state=SEED
)
Xf, Xs = Xr_train[fit_idx], Xr_train[stop_idx]
yf, ys = yr_train[fit_idx], yr_train[stop_idx]
wf = w_train[fit_idx] if w_train is not None else None
ws = w_train[stop_idx] if w_train is not None else None

XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=2.0,
    reg_alpha=0.1,
    objective="multi:softprob",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)
LGBM_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
CAT_PARAMS = dict(
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5,
    loss_function="MultiClass",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)
LOGIT_PARAMS = dict(max_iter=2000, C=1.0, solver="lbfgs", random_state=SEED, n_jobs=-1)


def fit_xgboost():
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS_CAP,
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS,
    )
    model.fit(
        Xf,
        yf,
        sample_weight=wf,
        eval_set=[(Xs, ys)],
        sample_weight_eval_set=[ws] if ws is not None else None,
        verbose=False,
    )
    return model, int(model.best_iteration)


def fit_lightgbm():
    model = LGBMClassifier(n_estimators=N_ESTIMATORS_CAP, **LGBM_PARAMS)
    model.fit(
        Xf,
        yf,
        sample_weight=wf,
        eval_set=[(Xs, ys)],
        eval_sample_weight=[ws] if ws is not None else None,
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    return model, int(model.best_iteration_)


def fit_catboost():
    model = CatBoostClassifier(iterations=N_ESTIMATORS_CAP, **CAT_PARAMS)
    model.fit(
        Xf,
        yf,
        sample_weight=wf,
        eval_set=(Xs, ys),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    return model, int(model.get_best_iteration())


def fit_logistic():
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**LOGIT_PARAMS))])
    model.fit(Xr_train, yr_train, clf__sample_weight=w_train)
    return model, None


BUILDERS = {
    "Logistic Regression": fit_logistic,
    "XGBoost": fit_xgboost,
    "LightGBM": fit_lightgbm,
    "CatBoost": fit_catboost,
}

fitted = {}
candidates = {}
for name, builder in BUILDERS.items():
    start = time.time()
    model, best_iter = builder()
    fitted[name] = model
    entry = {
        "best_iteration": best_iter,
        "train": evaluate(yr_train, model.predict(Xr_train).ravel(), model.predict_proba(Xr_train), CLASSES),
        "validation": evaluate(y_val.values, model.predict(Xt_val).ravel(), model.predict_proba(Xt_val), CLASSES),
        "fit_seconds": round(time.time() - start, 1),
    }
    entry["gaps"] = {
        "train_minus_val_accuracy": entry["train"]["accuracy"] - entry["validation"]["accuracy"],
        "train_minus_val_macro_f1": entry["train"]["macro_f1"] - entry["validation"]["macro_f1"],
    }
    candidates[name] = entry
    log(
        f"  {name:<20} valMacroF1={entry['validation']['macro_f1']:.4f} "
        f"valAcc={entry['validation']['accuracy']:.4f} "
        f"valHighRecall={entry['validation']['per_class']['High']['recall']:.4f} "
        f"trainF1-valF1={entry['gaps']['train_minus_val_macro_f1']:+.4f} iters={best_iter}"
    )

results["candidates"] = candidates
results["candidate_hyperparameters"] = {
    "XGBoost": {**XGB_PARAMS, "n_estimators_cap": N_ESTIMATORS_CAP, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
    "LightGBM": {**LGBM_PARAMS, "n_estimators_cap": N_ESTIMATORS_CAP, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
    "CatBoost": {**CAT_PARAMS, "iterations_cap": N_ESTIMATORS_CAP, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
    "Logistic Regression": LOGIT_PARAMS,
}

# ------------------------------- 5b. CatBoost with native categorical handling
log("Supplementary run: CatBoost with native categorical handling")
try:
    start = time.time()
    counts = np.bincount(y_train.values)
    native_weights = (
        (len(y_train) / (len(counts) * counts))[y_train.values]
        if BEST_STRATEGY == "class_weight_balanced"
        else None
    )
    nf_idx, ns_idx = train_test_split(
        np.arange(len(y_train)), test_size=EARLY_STOP_SIZE, stratify=y_train.values, random_state=SEED
    )
    native = CatBoostClassifier(iterations=N_ESTIMATORS_CAP, cat_features=CATEGORICAL, **CAT_PARAMS)
    native.fit(
        X_train.iloc[nf_idx],
        y_train.values[nf_idx],
        sample_weight=native_weights[nf_idx] if native_weights is not None else None,
        eval_set=(X_train.iloc[ns_idx], y_train.values[ns_idx]),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    results["catboost_native_categorical"] = {
        "note": (
            "identical split and imbalance strategy; categorical columns handed to CatBoost untouched "
            "so its ordered target statistics replace one-hot encoding"
        ),
        "applies_only_when_no_synthetic_rows": BEST_STRATEGY in ("none", "class_weight_balanced"),
        "best_iteration": int(native.get_best_iteration()),
        "validation": evaluate(
            y_val.values, native.predict(X_val).ravel(), native.predict_proba(X_val), CLASSES
        ),
        "fit_seconds": round(time.time() - start, 1),
    }
    log(
        "  CatBoost native cats valMacroF1="
        f"{results['catboost_native_categorical']['validation']['macro_f1']:.4f}"
    )
except Exception as error:  # noqa: BLE001
    results["catboost_native_categorical"] = {"status": "failed", "error": str(error)}
    log(f"  CatBoost native failed: {error}")

# --------------------------------------------------------------- 6. cross-validation
log(f"{CV_FOLDS}-fold stratified cross-validation on the training split only")
cv_X, cv_y = X_train, y_train
if len(cv_X) > CV_SUBSAMPLE:
    cv_X, _, cv_y, _ = train_test_split(
        X_train, y_train, train_size=CV_SUBSAMPLE, stratify=y_train, random_state=SEED
    )

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_results = {}
for name in BUILDERS:
    scores, high_recalls = [], []
    for fold_train, fold_val in skf.split(cv_X, cv_y):
        Xa, Xb = cv_X.iloc[fold_train], cv_X.iloc[fold_val]
        ya, yb = cv_y.iloc[fold_train].values, cv_y.iloc[fold_val].values

        prep = make_preprocessor().fit(Xa)
        Xa_enc, Xb_enc = prep.transform(Xa), prep.transform(Xb)

        Xa_enc, ya, fold_w = apply_strategy(
            Xa_enc, ya, BEST_STRATEGY, SEED, raw_frame=Xa, encoder=prep
        )

        if name == "Logistic Regression":
            m = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**LOGIT_PARAMS))])
            m.fit(Xa_enc, ya, clf__sample_weight=fold_w)
        elif name == "XGBoost":
            m = XGBClassifier(n_estimators=candidates["XGBoost"]["best_iteration"] + 1, **XGB_PARAMS)
            m.fit(Xa_enc, ya, sample_weight=fold_w, verbose=False)
        elif name == "LightGBM":
            m = LGBMClassifier(n_estimators=candidates["LightGBM"]["best_iteration"] + 1, **LGBM_PARAMS)
            m.fit(Xa_enc, ya, sample_weight=fold_w)
        else:
            m = CatBoostClassifier(iterations=candidates["CatBoost"]["best_iteration"] + 1, **CAT_PARAMS)
            m.fit(Xa_enc, ya, sample_weight=fold_w, verbose=False)

        pred = m.predict(Xb_enc).ravel()
        scores.append(float(f1_score(yb, pred, average="macro")))
        high_recalls.append(
            float(precision_recall_fscore_support(yb, pred, labels=[HIGH_INDEX], zero_division=0)[1][0])
        )

    cv_results[name] = {
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_std": float(np.std(scores, ddof=1)),
        "macro_f1_folds": scores,
        "high_recall_mean": float(np.mean(high_recalls)),
        "high_recall_std": float(np.std(high_recalls, ddof=1)),
    }
    log(f"  {name:<20} CV macroF1 = {np.mean(scores):.4f} +/- {np.std(scores, ddof=1):.4f}")

results["cross_validation"] = {
    "folds": CV_FOLDS,
    "rows_used": int(len(cv_X)),
    "note": (
        "stratified K-fold restricted to the training split; encoding and any resampling are refitted "
        "inside every fold; the test split is never involved"
    ),
    "results": cv_results,
}

# --------------------------------------------------------------- 7. selection
log("Selecting the final model")
ranked = sorted(
    candidates.items(),
    key=lambda kv: (
        round(kv[1]["validation"]["macro_f1"], 4),
        round(kv[1]["validation"]["per_class"]["High"]["recall"], 4),
        -abs(kv[1]["gaps"]["train_minus_val_macro_f1"]),
    ),
    reverse=True,
)
FINAL_NAME = ranked[0][0]
final_model = fitted[FINAL_NAME]

results["selection"] = {
    "rule": (
        "Rank candidates by validation Macro F1 rounded to four decimals; break ties on validation "
        "High-class recall and then on the smaller absolute train-validation Macro F1 gap. "
        "Cross-validation standard deviations are reported alongside so a reader can judge whether the "
        "winning margin exceeds run-to-run noise."
    ),
    "ranking": [
        {
            "model": name,
            "validation_macro_f1": entry["validation"]["macro_f1"],
            "validation_accuracy": entry["validation"]["accuracy"],
            "validation_high_recall": entry["validation"]["per_class"]["High"]["recall"],
            "validation_high_f1": entry["validation"]["per_class"]["High"]["f1"],
            "train_minus_val_macro_f1": entry["gaps"]["train_minus_val_macro_f1"],
            "cv_macro_f1_mean": cv_results[name]["macro_f1_mean"],
            "cv_macro_f1_std": cv_results[name]["macro_f1_std"],
        }
        for name, entry in ranked
    ],
    "final_model": FINAL_NAME,
    "margin_over_runner_up": (
        ranked[0][1]["validation"]["macro_f1"] - ranked[1][1]["validation"]["macro_f1"]
    ),
}
log(f"Final model: {FINAL_NAME}")

# ------------------------- 7b. High-class threshold study, tuned on validation only
val_proba_final = final_model.predict_proba(Xt_val)
threshold_rows = []
for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    masked = val_proba_final.copy()
    forced = masked[:, HIGH_INDEX] >= t
    alt = masked.copy()
    alt[:, HIGH_INDEX] = -1
    pred = np.where(forced, HIGH_INDEX, alt.argmax(axis=1))
    p, r, f, _ = precision_recall_fscore_support(
        y_val.values, pred, labels=[HIGH_INDEX], zero_division=0
    )
    threshold_rows.append(
        {
            "threshold": t,
            "high_precision": float(p[0]),
            "high_recall": float(r[0]),
            "high_f1": float(f[0]),
            "macro_f1": float(f1_score(y_val.values, pred, average="macro")),
            "accuracy": float(accuracy_score(y_val.values, pred)),
        }
    )

results["high_class_threshold_study"] = {
    "note": (
        "computed on the validation split only; the deployed model keeps the default argmax rule, so "
        "this table documents the operating points available if a farm wanted to trade precision for recall"
    ),
    "rows": threshold_rows,
}

# --------------------------------------------------------------- 8. final test
log("Single final evaluation on the untouched test split")
test_pred = final_model.predict(Xt_test).ravel()
test_proba = final_model.predict_proba(Xt_test)
test_metrics = evaluate(y_test.values, test_pred, test_proba, CLASSES)
results["final_test"] = test_metrics
results["test_set_classification_report"] = classification_report(
    y_test.values, test_pred, target_names=CLASSES, digits=4, zero_division=0
)
log(
    f"  test acc={test_metrics['accuracy']:.4f} macroF1={test_metrics['macro_f1']:.4f} "
    f"HighF1={test_metrics['per_class']['High']['f1']:.4f}"
)

# test-set metrics for every candidate, reported for transparency only; the model was
# already chosen from validation before any of these numbers existed
results["all_candidates_test"] = {
    name: evaluate(y_test.values, model.predict(Xt_test).ravel(), model.predict_proba(Xt_test), CLASSES)
    for name, model in fitted.items()
}

results["overfitting"] = {
    "final_model": FINAL_NAME,
    "train": {
        "accuracy": candidates[FINAL_NAME]["train"]["accuracy"],
        "macro_f1": candidates[FINAL_NAME]["train"]["macro_f1"],
    },
    "validation": {
        "accuracy": candidates[FINAL_NAME]["validation"]["accuracy"],
        "macro_f1": candidates[FINAL_NAME]["validation"]["macro_f1"],
    },
    "test": {"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"]},
    "gaps": {
        "train_minus_val_accuracy": candidates[FINAL_NAME]["train"]["accuracy"] - candidates[FINAL_NAME]["validation"]["accuracy"],
        "train_minus_test_accuracy": candidates[FINAL_NAME]["train"]["accuracy"] - test_metrics["accuracy"],
        "val_minus_test_accuracy": candidates[FINAL_NAME]["validation"]["accuracy"] - test_metrics["accuracy"],
        "train_minus_val_macro_f1": candidates[FINAL_NAME]["train"]["macro_f1"] - candidates[FINAL_NAME]["validation"]["macro_f1"],
        "train_minus_test_macro_f1": candidates[FINAL_NAME]["train"]["macro_f1"] - test_metrics["macro_f1"],
        "val_minus_test_macro_f1": candidates[FINAL_NAME]["validation"]["macro_f1"] - test_metrics["macro_f1"],
    },
    "controls_applied": [
        "early stopping against an inner split carved out of the training data, keeping validation free for selection",
        "shrinkage learning rate of 0.05",
        "row and column subsampling",
        "L1 and L2 penalties plus minimum-child-weight constraints",
        "bounded tree depth or leaf count",
        f"{CV_FOLDS}-fold stratified cross-validation to measure fold-to-fold variance",
    ],
}

# --------------------------------------------------------------- 9. importance
if hasattr(final_model, "feature_importances_"):
    importances = np.asarray(final_model.feature_importances_, dtype=float)
    total = importances.sum() or 1.0
    results["feature_importance"] = {
        name: float(value)
        for name, value in sorted(
            zip(FEATURE_NAMES, importances / total), key=lambda kv: kv[1], reverse=True
        )
    }

# --------------------------------------------------------------- 10. artifacts
log("Saving deployment artifacts")
deployment_pipeline = Pipeline([("preprocessor", tree_preprocessor), ("model", final_model)])
sanity = np.asarray(deployment_pipeline.predict(X_test.head(500))).ravel()
assert np.array_equal(sanity, test_pred[:500]), "pipeline output diverges from the evaluated model"

joblib.dump(deployment_pipeline, os.path.join(BASE_DIR, "irrigation_model.pkl"))

schema = {
    "final_model": FINAL_NAME,
    "target": TARGET,
    "class_labels": CLASSES,
    "class_index_to_label": {str(i): c for i, c in enumerate(CLASSES)},
    "raw_feature_order": X_raw.columns.tolist(),
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "categorical_levels": results["dataset"]["categorical_levels"],
    "numeric_ranges": results["dataset"]["numeric_ranges"],
    "encoded_feature_names": FEATURE_NAMES,
    "imbalance_strategy": BEST_STRATEGY,
    "random_seed": SEED,
    "training_rows": int(len(X_train)),
}
with open(os.path.join(BASE_DIR, "model_schema.json"), "w") as handle:
    json.dump(schema, handle, indent=2)

import sklearn  # noqa: E402
import xgboost  # noqa: E402
import lightgbm  # noqa: E402
import catboost  # noqa: E402
import imblearn  # noqa: E402
import sys  # noqa: E402

results["artifacts"] = {
    "irrigation_model.pkl": "sklearn Pipeline holding the training-fitted ColumnTransformer and the selected classifier",
    "model_schema.json": "raw feature order, categorical levels, numeric ranges and class mapping used by the API and the form",
    "model_metrics.json": "compact metrics file consumed by the Flask application",
    "experiments/results.json": "the complete experiment record behind every number in the documentation",
}
results["environment"] = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "lightgbm": lightgbm.__version__,
    "catboost": catboost.__version__,
    "imbalanced-learn": imblearn.__version__,
    "joblib": joblib.__version__,
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as handle:
    json.dump(results, handle, indent=2)

log("Done")
