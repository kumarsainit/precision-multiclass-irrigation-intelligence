# Precision Multi-Class Irrigation Intelligence

> **Live application:** https://precision-multiclass-irrigation.onrender.com/

Predicts whether a crop field needs **Low**, **Medium** or **High** irrigation from its soil,
weather, crop and field-management measurements, and serves that prediction through a Flask API and
a web dashboard.

Every number in this file is read from `experiments/results.json`, which is written by
`experiments/run_experiment.py` and by `notebook/train_model.ipynb`. Nothing here is typed in by
hand. Re-running the experiment regenerates the results file, the model artifact and the schema
together.

---

## Result summary

| | |
|---|---|
| Final model | **CatBoost** |
| Test accuracy | 98.55% |
| Test Macro F1 | 0.9709 |
| Test Weighted F1 | 0.9855 |
| Test Macro ROC-AUC | 0.9973 |
| Test High-class recall | 0.9219 |
| Test High-class F1 | 0.9409 |
| Imbalance strategy | `smote_original_partial`, training split only |
| Split | 70.0% / 15.0% / 15.0%, stratified, seed 42 |

The three gradient-boosting models finished within 0.00014 Macro F1 of each other, which is inside
the run-to-run noise. CatBoost is deployed because the pre-declared selection rule ranks it first
and it has the smallest train-to-validation gap, not because it is meaningfully better. See [Model
selection](#8-model-selection).

For the full write-up, including the methodology corrections and a viva question bank, read
[MAIN.md](MAIN.md).

---

## 1. Dataset

`notebook/train.csv` holds **630,000 rows** and **19 predictor columns** plus the target
`Irrigation_Need`. The `id` column is dropped before modelling because a row number carries no
agronomic information and would only invite leakage.

| Property | Value |
|---|---|
| Rows | 630,000 |
| Predictors | 19 (11 numeric, 8 categorical) |
| Missing values | 0 |
| Duplicate rows including target | 0 |
| Duplicate feature vectors | 0 |
| Identical feature vectors with different labels | 0 |

There are no missing values, no duplicate rows and no contradictory labels, so no imputation or
de-duplication step is required.

**Numeric columns:** `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`,
`Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`,
`Field_Area_hectare`, `Previous_Irrigation_mm`

**Categorical columns:** `Soil_Type`, `Crop_Type`, `Crop_Growth_Stage`, `Season`, `Irrigation_Type`,
`Water_Source`, `Mulching_Used`, `Region`

### Class distribution

| Class | Rows | Share |
|---|---:|---:|
| Low | 369,917 | 58.72% |
| Medium | 239,074 | 37.95% |
| High | 21,009 | 3.33% |
| **Total** | **630,000** | **100.00%** |

The majority-to-minority ratio is **17.61 to 1**. The High class, which is the one that matters most
operationally, is the rarest at 3.33% of the data. This is a severely imbalanced problem.

---

## 2. Leakage review

Every column in this dataset is a field measurement that a farmer or sensor would have before
deciding whether to irrigate. None of them is derived from the target, and none of them describes
the outcome of the irrigation decision. `Previous_Irrigation_mm` refers to past irrigation, which is
known at prediction time.

The checks that were run and their results:

- Feature vectors that appear more than once: **0**
- Identical feature vectors carrying different labels: **0**
- Row indices shared between train and validation: **0**
- Row indices shared between train and test: **0**
- Row indices shared between validation and test: **0**

---

## 3. Split strategy

The rows are split **before any preprocessing**, using a stratified split so each partition keeps
the same class proportions.

| Split | Rows | Share | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| Training | 441,000 | 70.0% | 258,941 | 167,352 | 14,707 |
| Validation | 94,500 | 15.0% | 55,488 | 35,861 | 3,151 |
| Test | 94,500 | 15.0% | 55,488 | 35,861 | 3,151 |

The order of operations is the point:

```
raw rows
  -> stratified 70 / 15 / 15 split
  -> fit the encoder on training rows only
  -> resample training rows only
  -> train
  -> compare models on validation
  -> score the test split once
```

The test split is never used to fit an encoder, generate synthetic rows, stop training early, tune a
threshold or choose a model.

---

## 4. Preprocessing

Categorical columns are one-hot encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')`
fitted on the training rows only, producing **35 encoded features**. `handle_unknown='ignore'` means
an unseen category at serving time is handled instead of crashing.

Numeric columns are left unscaled for the tree ensembles, which are invariant to monotonic
rescaling. The Logistic Regression baseline has a `StandardScaler` inside its own pipeline, because
a linear model genuinely needs comparable scales to converge. That is a requirement of the
algorithm, not an advantage handed to it.

---

## 5. Imbalance handling

Eight strategies were compared under one fixed probe model (LightGBM with fixed hyperparameters),
one fixed split and one fixed metric. All numbers below are on the validation split.

| Strategy | Training rows after | Macro F1 | Accuracy | High precision | High recall | High F1 |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 441,000 | 0.9703 | 0.9844 | 0.9700 | 0.9146 | 0.9415 |
| `class_weight_balanced` | 441,000 | 0.9640 | 0.9830 | 0.9123 | 0.9378 | 0.9249 |
| `random_oversampling` | 776,823 | 0.9676 | 0.9844 | 0.9248 | 0.9410 | 0.9328 |
| `random_undersampling` | 44,121 | 0.9478 | 0.9794 | 0.8176 | 0.9559 | 0.8813 |
| `smote_on_encoded` | 776,823 | 0.9702 | 0.9845 | 0.9625 | 0.9207 | 0.9411 |
| `smote_original_partial` **(selected)** | 558,941 | 0.9709 | 0.9846 | 0.9667 | 0.9207 | 0.9431 |
| `smotenc_on_raw` | 776,823 | 0.9636 | 0.9826 | 0.9156 | 0.9330 | 0.9242 |
| `smotetomek_on_encoded` | 747,895 | 0.9698 | 0.9843 | 0.9606 | 0.9203 | 0.9400 |

### How the strategy was chosen

The gaps between the top strategies are small, so the rule accounts for noise rather than picking
the top row blindly. The probe was repeated across five seeds to measure how much validation Macro
F1 moves for reasons unrelated to the strategy:

| Strategy | Mean Macro F1 | Standard deviation | Range across seeds |
|---|---:|---:|---:|
| `none` | 0.9703 | 0.0003 | 0.0007 |
| `smote_original_partial` | 0.9707 | 0.0004 | 0.0010 |

Best validation Macro F1 was **0.9709**, and one noise standard deviation below that is **0.9705**.
Only `smote_original_partial` clears that bar, so it is the selected strategy. Had several
strategies cleared it, the rule was to take the one imposing the fewest assumptions.

### What this shows

- **SMOTE genuinely helps here, but only slightly.** Partial SMOTE beats no resampling by 0.0006
  Macro F1. That is real but small, and no resampling would have been an acceptable choice.
- **SMOTENC, the variant designed for mixed data, performs worse** (0.9636) than plain SMOTE on the
  one-hot matrix. This is the opposite of the textbook expectation and is worth stating openly.
- **Aggressive rebalancing trades precision for recall.** Random undersampling reaches the best High
  recall of any strategy (0.9559) but collapses High precision to 0.8176, and discards 396,879
  training rows.

The chosen strategy is applied to the training split only, after the split has already happened.

---

## 6. Models compared

Four models, one protocol. Each sees the same training rows, the same encoding and the same
imbalance strategy. Early stopping runs against an inner slice carved out of the **training** split,
so the validation split stays reserved for model comparison.

| Model | Validation Macro F1 | Cross-validated Macro F1 | Validation High recall | Train minus validation Macro F1 |
|---|---:|---:|---:|---:|
| CatBoost | 0.9706 | 0.9680 ± 0.0012 | 0.9191 | +0.0126 |
| LightGBM | 0.9704 | 0.9691 ± 0.0019 | 0.9184 | +0.0235 |
| XGBoost | 0.9703 | 0.9681 ± 0.0020 | 0.9156 | +0.0183 |
| Logistic Regression | 0.7796 | 0.7805 ± 0.0045 | 0.8248 | +0.0763 |

Cross-validation is 5-fold stratified on 150,000 training rows, with the encoder and the resampler
refitted inside every fold. The test split takes no part in it.

CatBoost was also run with its **native categorical handling** instead of one-hot encoding, scoring
0.9701 validation Macro F1 against 0.9706 with the shared encoding. Native handling did not help on
this dataset, so the shared encoding was kept for every model.

---

## 7. Test set results

Scored once, after model selection was complete.

| Metric | Value |
|---|---:|
| Accuracy | 0.9855 |
| Macro precision | 0.9777 |
| Macro recall | 0.9644 |
| Macro F1 | 0.9709 |
| Weighted F1 | 0.9855 |
| Macro ROC-AUC | 0.9973 |
| Macro average precision | 0.9870 |
| Micro average precision | 0.9965 |

### Class-wise

| Class | Precision | Recall | F1 | Test rows |
|---|---:|---:|---:|---:|
| High | 0.9606 | 0.9219 | 0.9409 | 3,151 |
| Medium | 0.9856 | 0.9760 | 0.9808 | 35,861 |
| Low | 0.9868 | 0.9952 | 0.9910 | 55,488 |

### Confusion matrix

| | Predicted High | Predicted Low | Predicted Medium |
|---|---:|---:|---:|
| **True High** | 2,905 | 0 | 246 |
| **True Low** | 0 | 55,223 | 265 |
| **True Medium** | 119 | 740 | 35,002 |

Two observations matter operationally:

1. **No High-need field was ever predicted Low.** All 246 High-class errors land on Medium, which is
   a one-step error rather than a total miss.
2. **The largest single error cell is 740 Medium fields predicted as Low.** Under-watering a Medium
   field is the most common practical failure of this model.

### All models on the test split

Reported for transparency. Selection was already finished before these were computed.

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC | Micro AP | High F1 |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.9855 | 0.9709 | 0.9855 | 0.9973 | 0.9965 | 0.9409 |
| XGBoost | 0.9855 | 0.9708 | 0.9854 | 0.9976 | 0.9970 | 0.9407 |
| LightGBM | 0.9851 | 0.9704 | 0.9850 | 0.9976 | 0.9970 | 0.9402 |
| Logistic Regression | 0.8631 | 0.7848 | 0.8656 | 0.9618 | 0.9439 | 0.6268 |

---

## 8. Model selection

**Rule, fixed before the comparison:** Rank candidates by validation Macro F1 rounded to four
decimals; break ties on validation High-class recall and then on the smaller absolute
train-validation Macro F1 gap. Cross-validation standard deviations are reported alongside so a
reader can judge whether the winning margin exceeds run-to-run noise.

CatBoost finished first, ahead of the runner-up by **0.00014** validation Macro F1. That margin is
smaller than the seed-to-seed noise measured above and smaller than the cross-validation fold spread
of every model in the table.

**The honest reading:** CatBoost, LightGBM and XGBoost are statistically indistinguishable on this
dataset. Cross-validation actually ranks LightGBM highest (0.9691) while the single validation split
ranks CatBoost highest. CatBoost is deployed because the rule was declared in advance and ranks it
first, and because it carries the smallest train-to-validation gap (+0.0126 against +0.0183 for
XGBoost and +0.0235 for LightGBM). Deploying either of the other two would have been defensible.

Logistic Regression is a different story. At 0.7796 validation Macro F1 and 0.4926 High-class
precision it is decisively worse, which is the useful result: the class boundaries in this dataset
are genuinely non-linear and a linear model cannot represent them.

---

## 9. Overfitting analysis

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Training | 0.9843 | 0.9831 |
| Validation | 0.9849 | 0.9706 |
| Test | 0.9855 | 0.9709 |

| Gap | Accuracy | Macro F1 |
|---|---:|---:|
| Train minus validation | -0.0005 | +0.0126 |
| Train minus test | -0.0012 | +0.0123 |
| Validation minus test | -0.0006 | -0.0003 |

**Test accuracy is marginally higher than training accuracy**, and validation and test agree to
0.0003 Macro F1. A memorising model does not behave this way.

The Macro F1 training gap of +0.0123 is not evidence of overfitting either: training Macro F1 is
measured on the resampled training set, where the High class makes up about 18% of rows instead of
3.3%, and Macro F1 is sensitive to that change of mix. Accuracy, which is measured the same way on
both, shows no gap at all.

Controls in place: early stopping against an inner training slice, learning rate 0.05, bounded
depth, L2 leaf regularisation, and cross-validation to measure fold-to-fold variance.

**Why the scores are high.** The honest explanation is not that the model is exceptional but that
the dataset is easy. Class membership is driven almost entirely by a handful of columns with clean
separation: mean soil moisture is 17.7% for High against 43.3% for Low, and mean temperature is 34.6
against 25.3 degrees Celsius. A leakage-free pipeline with no resampling at all already reaches
0.9703 Macro F1. This strongly suggests the labels were generated from a rule over these features,
which is a property of the dataset and a real limitation of the project. It is discussed in
[MAIN.md](MAIN.md#20-limitations).

---

## 10. Feature importance

| Rank | Feature | Share of total importance |
|---:|---|---:|
| 1 | `Soil_Moisture` | 0.2351 |
| 2 | `Crop_Growth_Stage_Harvest` | 0.1328 |
| 3 | `Crop_Growth_Stage_Sowing` | 0.1163 |
| 4 | `Mulching_Used_Yes` | 0.1052 |
| 5 | `Temperature_C` | 0.1019 |
| 6 | `Wind_Speed_kmh` | 0.0939 |
| 7 | `Rainfall_mm` | 0.0707 |
| 8 | `Crop_Growth_Stage_Vegetative` | 0.0137 |
| 9 | `Humidity` | 0.0124 |
| 10 | `Water_Source_River` | 0.0102 |
| 11 | `Previous_Irrigation_mm` | 0.0098 |
| 12 | `Water_Source_Reservoir` | 0.0088 |

Soil moisture dominates, followed by growth stage, mulching, temperature, wind speed and rainfall.
These are the same columns that show the largest class separation in the raw data, so the model is
relying on agronomically sensible signals rather than an artefact.

---

## 11. High-class decision threshold

The deployed model uses the ordinary rule: predict whichever class has the highest probability. This
table, computed on the **validation** split, documents the operating points available if a farm
preferred to catch more High-need fields at the cost of more false alarms.

| Threshold for predicting High | High precision | High recall | High F1 | Macro F1 |
|---:|---:|---:|---:|---:|
| 0.2 | 0.9186 | 0.9378 | 0.9281 | 0.9657 |
| 0.3 | 0.9483 | 0.9308 | 0.9395 | 0.9699 |
| 0.4 | 0.9595 | 0.9248 | 0.9418 | 0.9707 |
| 0.5 | 0.9647 | 0.9191 | 0.9413 | 0.9706 |
| 0.6 | 0.9693 | 0.9111 | 0.9393 | 0.9699 |
| 0.7 | 0.9730 | 0.9042 | 0.9373 | 0.9692 |
| 0.8 | 0.9795 | 0.8943 | 0.9350 | 0.9683 |

The default of 0.5 is close to optimal on Macro F1, so no threshold override is applied. This
analysis is deliberately on validation, not test.

---

## 12. Deployment

```
raw field values (19 columns)
        |
        v
   Flask /api/predict
        |
        v
   schema validation against model_schema.json
        |
        v
   irrigation_model.pkl
   (fitted OneHotEncoder + CatBoost)
        |
        v
   Low / Medium / High + class probabilities
```

The model artifact is a single scikit-learn `Pipeline` containing the encoder that was fitted during
training and the classifier. The API therefore hands the pipeline a row of **raw** field values and
the artifact applies exactly the encoding it was fitted with. There is no separate encoding step in
the web application that could drift away from training.

`experiments/verify_deployment.py` proves this: it rebuilds the split, scores the test rows through
the saved artifact, checks the accuracy, Macro F1 and confusion matrix against
`experiments/results.json`, then sends raw rows through the live HTTP API and requires identical
labels and probabilities.

### API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Prediction dashboard |
| `/models` | GET | Model specification and full experiment record |
| `/history` | GET | Session prediction log |
| `/api/predict` | POST | Prediction from raw field values |
| `/api/models` | GET | All candidate metrics and the final model |
| `/api/schema` | GET | Feature schema: columns, categorical levels, numeric ranges |
| `/api/history` | GET | Recent predictions |

`/api/predict` accepts either the exact training column names (`Soil_Moisture`) or lower-case
aliases (`soil_moisture`, `windspeed_kmph`). It rejects unknown categorical values with the list of
allowed levels rather than silently scoring a field the model has never seen.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"Soil_Type":"Sandy","Soil_pH":6.5,"Soil_Moisture":15.0,"Organic_Carbon":0.9,
       "Electrical_Conductivity":1.7,"Temperature_C":38.0,"Humidity":35.0,"Rainfall_mm":400.0,
       "Sunlight_Hours":10.0,"Wind_Speed_kmh":18.0,"Crop_Type":"Maize",
       "Crop_Growth_Stage":"Flowering","Season":"Zaid","Irrigation_Type":"Canal",
       "Water_Source":"River","Field_Area_hectare":5.0,"Mulching_Used":"No",
       "Previous_Irrigation_mm":40.0,"Region":"North"}'
```

---

## 13. Repository layout

```
.
|-- app.py                          Flask application
|-- irrigation_model.pkl            fitted encoder + CatBoost
|-- model_schema.json               feature schema and class mapping
|-- requirements.txt                runtime dependencies
|-- requirements-dev.txt            experiment dependencies
|-- Procfile                        gunicorn entry point
|-- .python-version                 3.13.9
|-- README.md
|-- MAIN.md                         full research and defence document
|-- experiments/
|   |-- run_experiment.py           the experiment, headless
|   |-- verify_deployment.py        train-serving consistency check
|   `-- results.json                the complete experiment record
|-- notebook/
|   |-- train.csv
|   `-- train_model.ipynb           the same experiment, with plots
|-- templates/
`-- static/
```

`experiments/run_experiment.py` and `notebook/train_model.ipynb` run the same code. The notebook
cells are generated from the script, so they cannot disagree about what the experiment did.

---

## 14. Running it

```bash
pip install -r requirements-dev.txt
python3 experiments/run_experiment.py
python3 app.py
```

`run_experiment.py` takes roughly 40 minutes on a laptop; SMOTENC over 441,000 rows is the slow
part. It writes `irrigation_model.pkl`, `model_schema.json` and `experiments/results.json`.

To verify a deployment:

```bash
python3 app.py &
python3 experiments/verify_deployment.py
```

Production:

```bash
gunicorn app:app
```

---

## 15. Reproducibility

Random seed **42** fixes the split, the resampler and every model. The class mapping is recorded in
`model_schema.json` as {"0": "High", "1": "Low", "2": "Medium"} and read from there by the
application rather than hardcoded.

| Package | Version |
|---|---|
| python | 3.13.9 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| catboost | 1.2.10 |
| imbalanced-learn | 0.14.0 |
| joblib | 1.5.2 |

---

## 16. Known limitations

- The dataset behaves as though it was generated from a rule over a few columns. High scores here do
  not imply the same performance on real field data.
- No temporal structure. Rows are treated as independent, so the model cannot use irrigation history
  as a sequence.
- No live sensor integration; every value is entered by hand or supplied by the caller.
- Region is a coarse label with no soil-map, elevation or evapotranspiration data behind it.
- The model outputs a class, not a water volume in millimetres, because the dataset contains no such
  target.

These are discussed at length in [MAIN.md](MAIN.md#20-limitations).
