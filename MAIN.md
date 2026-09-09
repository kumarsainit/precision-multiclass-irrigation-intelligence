# Precision Multi-Class Irrigation Intelligence

A complete account of what this project does, how it was built, what was wrong with the earlier
version, and how each decision can be defended.

Every figure quoted here comes from `experiments/results.json`, produced by one run of
`experiments/run_experiment.py` (equivalently, `notebook/train_model.ipynb`). No number in this
document was typed in by hand.

---

## 1. Project objective

### What

Given the measurable condition of a crop field, decide how much irrigation it needs, expressed as
one of three levels: **Low**, **Medium** or **High**.

### Why it matters

Irrigation is the single largest use of fresh water in agriculture. Two mistakes cost money in
opposite directions. Watering a field that does not need it wastes water, fuel and pumping time, and
can waterlog the root zone. Failing to water a field that does need it costs yield, and at a
water-sensitive growth stage such as flowering it can cost a large fraction of the season's output.

A farmer already knows most of the inputs: how wet the soil is, how hot it has been, what stage the
crop is at, whether the field is mulched. What is hard is combining fifteen or twenty such readings
consistently across many fields, every week, without fatigue or guesswork. That combination is what
a classifier can do reliably.

### Why three classes instead of a number

The dataset records a decision level, not a volume of water, so a three-class label is what can
honestly be learned from it. Three levels also match how the decision is actually made in practice:
skip, normal dose, or priority. A model that outputs "42.7 mm" would look more precise than the
underlying data supports.

### Result

A CatBoost classifier that reaches 98.55% accuracy and 0.9709 Macro F1 on a test split it never saw
during training or tuning, with 0.9219 recall on the critical High class.

### What we learned

The hardest part of this project was never the modelling. It was making sure the evaluation deserved
to be believed.

---

## 2. Dataset

### What

`notebook/train.csv`: **630,000 rows**, **19 predictor columns**, one target column
`Irrigation_Need`.

**The 11 numeric columns**

| Column | Meaning | Range in the data |
|---|---|---|
| `Soil_Moisture` | volumetric root-zone moisture, percent | 8.00 to 64.99 |
| `Temperature_C` | ambient temperature | 12.00 to 42.00 |
| `Humidity` | relative humidity, percent | 25.00 to 94.99 |
| `Rainfall_mm` | cumulative rainfall | 0.38 to 2499.69 |
| `Wind_Speed_kmh` | wind speed | 0.50 to 20.00 |
| `Sunlight_Hours` | daily sunlight | 4.00 to 11.00 |
| `Soil_pH` | soil acidity | 4.80 to 8.20 |
| `Organic_Carbon` | soil organic carbon, percent | 0.30 to 1.60 |
| `Electrical_Conductivity` | soil salinity proxy | 0.10 to 3.50 |
| `Field_Area_hectare` | field size | 0.30 to 15.00 |
| `Previous_Irrigation_mm` | water applied previously | 0.02 to 119.99 |

**The 8 categorical columns**

| Column | Levels |
|---|---|
| `Soil_Type` | Clay, Loamy, Sandy, Silt |
| `Crop_Type` | Cotton, Maize, Potato, Rice, Sugarcane, Wheat |
| `Crop_Growth_Stage` | Flowering, Harvest, Sowing, Vegetative |
| `Season` | Kharif, Rabi, Zaid |
| `Irrigation_Type` | Canal, Drip, Rainfed, Sprinkler |
| `Water_Source` | Groundwater, Rainwater, Reservoir, River |
| `Mulching_Used` | No, Yes |
| `Region` | Central, East, North, South, West |

### Data quality

| Check | Result |
|---|---|
| Missing values anywhere | 0 |
| Rows duplicated including the target | 0 |
| Feature vectors appearing more than once | 0 |
| Identical feature vectors with different labels | 0 |

The dataset is unusually clean. Nothing needed imputing, de-duplicating or repairing, which is
itself a hint about how it was produced.

### The classes

| Class | Rows | Share |
|---|---:|---:|
| Low | 369,917 | 58.72% |
| Medium | 239,074 | 37.95% |
| High | 21,009 | 3.33% |

### What we learned

The `id` column was dropped immediately. It is a row number, it correlates with nothing real, and if
a tree ever split on it the model would be learning the order of the file rather than agronomy.

---

## 3. The class imbalance problem

### What

The three classes are not equally represented. Low has 369,917 rows and High has 21,009. The ratio
between the largest and smallest class is **17.61 to 1**.

### Why it matters

A model is trained by minimising average error. If 58.7% of rows are Low, then a model that ignores
the other two classes entirely and predicts Low every single time is already right 58.7% of the
time. It has learned nothing, and it would score 58.7% accuracy.

Worse, the class it would ignore is the one that matters most. High is the situation where a field
is about to suffer water stress. A model that quietly never predicts High is worse than useless: it
is confidently wrong exactly when being right has value.

### The concrete danger

Consider the 3,151 genuinely High-need fields in the test split. A majority-class predictor would
miss all 3,151 of them while still reporting 58.7% accuracy. That is why accuracy alone cannot be
the headline metric for this project, and why the class-wise recall on High is quoted everywhere
alongside it.

### How the imbalance was detected

By counting. `value_counts()` on the target column, then the ratio between the largest and smallest
class, then the same counts recomputed inside each of the three splits to confirm the stratified
split preserved the proportions.

### Is the imbalance meaningful, or an artefact?

It is meaningful. High irrigation need is genuinely a rare event: most fields on most days are fine.
An irrigation dataset in which a third of fields were in crisis would be the suspicious one. The
imbalance should therefore be handled, not corrected away as if it were a data-collection error, and
the test split must keep the natural proportions so the reported numbers describe real conditions.

### What we learned

Imbalance is not a defect to be erased. It is a property of the problem, and the job is to stop it
from biasing the training signal while leaving the evaluation honest.

---

## 4. What was wrong with the earlier version

This section is deliberately blunt. The earlier version of this project reported 0.9853 accuracy and
0.9704 Macro F1 with XGBoost. Those numbers were not fabricated, and as it turns out they were not
far off. But the methodology behind them had defects that a panel is entitled to press on, and three
of them were real.

### Problem 1: encoding was fitted on the whole dataset before splitting

**What was done.** `pd.get_dummies(features, drop_first=True)` was applied to all 630,000 rows, and
only afterwards were the rows split into train, validation and test.

**Why it is a problem.** The set of one-hot columns, and which level got dropped as the reference,
was decided by looking at every row including the test rows. This is a mild form of leakage. Its
practical effect on this dataset is close to zero, because every categorical level appears many
times in every split, so the columns would have come out identical either way. But the *pattern* is
wrong, and on a dataset with a rare category it would produce a column that exists at training time
and silently vanishes at serving time.

**How it was fixed.** The rows are split first. A `OneHotEncoder` is then fitted on the training
rows only, with `handle_unknown='ignore'` so an unseen category at serving time is handled rather
than crashing the request.

### Problem 2: the decision threshold was tuned on the test set

**What was done.** After selecting XGBoost, the notebook swept the High-class probability threshold
across 0.3, 0.4, 0.5, 0.6, 0.7 and 0.8 and displayed the resulting confusion matrices **on the test
set**.

**Why it is a problem.** This is the clearest defect of the three. The test set exists to give one
unbiased estimate of performance on unseen data. Once you look at it repeatedly to pick a setting,
it has become a second validation set and the estimate it produces is optimistic. Even if the
threshold was never actually changed, the analysis was performed in the wrong place.

**How it was fixed.** The same sweep is now run on the **validation** split. The result is recorded
as documentation of the available operating points, and the deployed model keeps the ordinary
highest-probability rule.

### Problem 3: SMOTE was interpolating one-hot columns

**What was done.** SMOTE was applied to the one-hot encoded matrix, so it interpolated between
neighbouring rows across all 35 columns including the binary indicators.

**Why it is a problem.** Interpolating soil moisture between two similar fields is defensible: a
field halfway between 30% and 34% moisture is a plausible field. Interpolating `Soil_Type_Loamy`
between 0 and 1 is not. It produces a synthetic row that is 0.4 Loamy and 0.6 Sandy, which is not a
soil type that exists. Tree models tolerate this because they can split at 0.5, but the synthetic
rows are not valid fields and the practice is hard to defend.

**How it was investigated.** SMOTENC, the variant designed for mixed numeric and categorical data,
was measured against plain SMOTE under identical conditions. It interpolates the numeric columns and
assigns each categorical column by a majority vote among the neighbours, so every synthetic row
stays a real, valid field.

**What was found, and this is the interesting part.** SMOTENC performed **worse**: 0.9636 validation
Macro F1 against 0.9702 for plain SMOTE on the one-hot matrix. The theoretically cleaner method
lost. That result is reported as found rather than buried, and the practical explanation is
discussed in section 6.

### What was checked and found to be fine

Two things the earlier version was accused of, which the audit cleared:

- **SMOTE was applied only to the training split.** The earlier code did this correctly. Validation
  and test were untouched, and the notebook asserted zero index overlap between splits. That part
  was sound.
- **The reported metrics were not inflated by leakage.** A completely clean pipeline with no
  resampling at all reaches 0.9703 Macro F1. The old numbers were roughly right. They were high
  because the dataset is easy, not because the evaluation was broken.

### One more defect, in the deployed application rather than the model

The earlier web form offered crop types (Soybean, Tomato, Onion, Sunflower), growth stages
(Germination, Seedling, Fruit Development, Maturity) and water sources (Canal, Borewell, Drip
System, Sprinkler) **that do not exist in the training data**. All four water-source options were
invalid. A user selecting any of them produced an all-zero one-hot block, so the model silently
scored the field as if it were the reference category. The form also labelled `Rainfall_mm` as
"Recent Rainfall" and defaulted it to 2.0 mm, when the training column is cumulative rainfall
ranging from 0 to 2500 mm with a median of 1467. And four features (`Soil_Type`, `Season`,
`Irrigation_Type`, `Region`) were hardcoded to fixed defaults, so the user could not set them at
all.

This is train-serving skew, and it was arguably more damaging than any of the modelling issues,
because it meant live predictions were being made on inputs unlike anything the model was trained
on. Section 18 describes the fix.

### What we learned

Reporting a mistake honestly is stronger than hiding it. Two of these three modelling defects had
little numerical effect, and saying so is more credible than either pretending they were
catastrophic or pretending they never happened. The serving defect, which nobody had flagged, was
the one that actually mattered.

---

## 5. Correct data splitting

### What

The 630,000 rows are divided into three groups, once, before anything else happens:

| Split | Rows | Share | Purpose |
|---|---:|---:|---|
| Training | 441,000 | 70.0% | the model learns from these |
| Validation | 94,500 | 15.0% | compare models and settings |
| Test | 94,500 | 15.0% | scored once, at the very end |

The split is **stratified**, meaning each group keeps the same class proportions as the full
dataset:

| Split | Low | Medium | High |
|---|---:|---:|---:|
| Training | 258,941 | 167,352 | 14,707 |
| Validation | 55,488 | 35,861 | 3,151 |
| Test | 55,488 | 35,861 | 3,151 |

### Why the test set must stay untouched

An analogy that survives contact with a viva panel: the training set is the textbook, the validation
set is the practice papers, and the test set is the final exam. You may work through the textbook as
many times as you like. You may use the practice papers to decide which techniques to rely on. But
if you see the final exam in advance, your score stops measuring what you know and starts measuring
how well you memorised that particular paper.

Concretely, the test split is never used to fit the encoder, to generate synthetic rows, to decide
when to stop training, to tune a threshold, to select features, or to choose between models. It is
opened exactly once, in section 14 of the notebook, after every decision is already fixed.

### An extra precaution: early stopping does not touch validation

Gradient boosting needs to know when to stop adding trees. The usual approach is to watch
performance on the validation set and stop when it stops improving. That works, but it means the
validation set has influenced the model, and then using the same validation set to choose between
models is slightly circular.

So early stopping here runs against an **inner slice carved out of the training split** (15% of the
training rows). The validation split is reserved purely for comparing finished models.

### Verification

| Check | Result |
|---|---|
| Row indices shared between train and validation | 0 |
| Row indices shared between train and test | 0 |
| Row indices shared between validation and test | 0 |

These are asserted in the code, so the notebook stops if any of them is ever non-zero.

### Why 70 / 15 / 15 was kept

The earlier project used this split and there was no methodological reason to change it. With
630,000 rows, 15.0% still gives 94,500 validation rows including 3,151 High-class examples, which is
far more than enough for a stable estimate. Changing a split after seeing results is exactly the
kind of decision that invites suspicion, so it was left alone.

### What we learned

The order of operations is the whole game. Split, then fit, then resample, then train, then compare,
then test. Every methodological problem found in the earlier version was an operation performed at
the wrong point in that sequence.

---

## 6. Correct imbalance handling

### What was compared

Eight strategies, each trained with the same probe model (LightGBM with fixed hyperparameters) on
the same training split, each scored on the same validation split.

| Strategy | Training rows after | Macro F1 | High precision | High recall | High F1 |
|---|---:|---:|---:|---:|---:|
| **none** | 441,000 | 0.9703 | 0.9700 | 0.9146 | 0.9415 |
| **class_weight_balanced** | 441,000 | 0.9640 | 0.9123 | 0.9378 | 0.9249 |
| **random_oversampling** | 776,823 | 0.9676 | 0.9248 | 0.9410 | 0.9328 |
| **random_undersampling** | 44,121 | 0.9478 | 0.8176 | 0.9559 | 0.8813 |
| **smote_on_encoded** | 776,823 | 0.9702 | 0.9625 | 0.9207 | 0.9411 |
| **smote_original_partial** (selected) | 558,941 | 0.9709 | 0.9667 | 0.9207 | 0.9431 |
| **smotenc_on_raw** | 776,823 | 0.9636 | 0.9156 | 0.9330 | 0.9242 |
| **smotetomek_on_encoded** | 747,895 | 0.9698 | 0.9606 | 0.9203 | 0.9400 |

### What each strategy is, in plain terms

- **`none`** - train on the data exactly as it is. Simplest possible choice, no assumptions.
- **`class_weight_balanced`** - keep every row, but tell the model that a mistake on a rare class
  counts for more. No synthetic data at all.
- **`random_oversampling`** - copy existing minority rows until every class is the same size. Adds
  no new information, only repetition, and risks the model memorising the repeated rows.
- **`random_undersampling`** - throw away majority rows until every class is the same size. Here
  that meant discarding 396,879 of 441,000 training rows.
- **`smote_on_encoded`** - Synthetic Minority Over-sampling Technique. For each minority row, find
  its five nearest minority neighbours, pick one, and create a new point somewhere on the line
  between them. Applied here to the one-hot matrix, so it interpolates the indicator columns too.
- **`smote_original_partial`** - the same, but raising High to 100,000 rows and Medium to 200,000
  instead of matching the majority. Partial rebalancing rather than full equalisation.
- **`smotenc_on_raw`** - SMOTE for mixed data. Numeric columns are interpolated; each categorical
  column is set by a majority vote among the neighbours, so every synthetic row is a valid field.
- **`smotetomek_on_encoded`** - SMOTE, then remove Tomek links: pairs of nearest neighbours
  belonging to different classes, which sit right on the boundary. The idea is to clean up an
  ambiguous frontier.

### How the choice was made

The top strategies are separated by tiny margins, so picking the highest number blindly would be
picking noise. The rule, fixed in advance:

1. Take the best validation Macro F1.
2. Measure how much that metric moves across five random seeds, for reasons unrelated to the
   strategy.
3. Treat everything within one standard deviation of the best as indistinguishable.
4. Among those, choose the strategy that assumes the least about the data.

The noise measurement:

| Strategy | Mean Macro F1 over 5 seeds | Standard deviation | Spread |
|---|---:|---:|---:|
| `none` | 0.9703 | 0.0003 | 0.0007 |
| `smote_original_partial` | 0.9707 | 0.0004 | 0.0010 |

Best Macro F1 was 0.9709; one standard deviation below is 0.9705. Only `smote_original_partial`
clears that bar, so it is selected on its own merits rather than by a tie-break.

### Why the alternatives were rejected

- **`none`** at 0.9703 is genuinely close, and would have been an acceptable choice. It lost by
  0.0006, which is larger than the noise floor but small in absolute terms.
- **`class_weight_balanced`** (0.9640) and **`random_oversampling`** (0.9676) both raise High recall
  but drop High precision by more, so Macro F1 falls.
- **`random_undersampling`** (0.9478) has the best High recall of any strategy at 0.9559, but High
  precision collapses to 0.8176: roughly one in five High alerts would be a false alarm. It also
  throws away most of the training data.
- **`smotenc_on_raw`** (0.9636) is the theoretically correct choice for mixed data and lost anyway.
  The likely reason is that majority-vote categorical assignment collapses the diversity of the
  synthetic rows: many neighbours share the same common category, so the synthetic High rows end up
  clustered in a few categorical combinations rather than spread across the space. One-hot
  interpolation, though it produces impossible fractional values, at least preserves variety, and
  tree models split those fractions harmlessly at 0.5.
- **`smotetomek_on_encoded`** (0.9698) removed 28,928 borderline rows and did not improve on plain
  SMOTE. The class boundaries here are clean enough that there is little ambiguity to clean up.

### Where it is applied, and why that matters

Only on the 441,000 training rows, after the split. It raised High from 14,707 to 100,000 and Medium
from 167,352 to 200,000, leaving Low at 258,941.

If synthetic rows reached validation or test, the model would be scored partly on data invented from
rows it had already learned. Since a synthetic point sits between two real training points, the
model has effectively seen it. Scores would rise and mean nothing. Keeping resampling inside the
training split is what makes the reported 98.55% an estimate of real-world performance rather than a
measure of how well the model reproduces its own training data.

### What we learned

The strategy that theory recommends is not automatically the strategy that works. Measuring eight
options under one protocol took an hour of compute and produced a defensible answer, which is a much
better position than asserting that SMOTE is standard practice.

---

## 7. Feature engineering

### What

Deliberately minimal. No new features were invented, no columns were dropped except `id`, and no
transformations were applied beyond what each model requires.

**Categorical encoding.** The 8 categorical columns become 24 binary indicator columns via
`OneHotEncoder(drop='first', handle_unknown='ignore')`, giving 35 features in total.

- `drop='first'` removes one level per column as a reference. Four soil types need only three
  columns; the fourth is implied when all three are zero. This avoids a redundant column that
  carries no extra information.
- `handle_unknown='ignore'` means a category the encoder never saw produces an all-zero block
  instead of an exception. The API also validates categories against the schema before this point,
  so an unknown value is reported to the caller rather than silently scored.

**Numeric handling.** Left unscaled for the tree ensembles. A decision tree asks "is soil moisture
below 25?", and the answer does not change if every value is divided by ten, so scaling would be a
no-op. Logistic Regression gets a `StandardScaler` inside its own pipeline because gradient descent
on unscaled features converges poorly when one column ranges to 2,500 and another to 1.6.

**Why no engineered features.** Combinations such as a temperature-times-wind evaporation index were
considered and left out. Gradient-boosted trees construct interactions internally by splitting on
one feature and then another, so hand-built products mostly add correlated columns. Adding them
would also have made the comparison against the linear baseline less informative.

### Feature schema

The exact column order, the categorical levels and the numeric ranges are saved to
`model_schema.json` at training time. The API and the web form both read from that file, so the
inputs the application accepts are defined by the training data rather than by whatever was typed
into an HTML template.

### What we learned

Every preprocessing step is fitted inside the pipeline object that gets deployed. That is what makes
"the training input equals the deployment input" a checkable claim rather than an intention.

---

## 8. Models evaluated

Four models were trained under one identical protocol. Each saw the same training rows, the same
encoding and the same imbalance strategy, so any difference between them is a difference between the
algorithms rather than a difference in how they were treated.

### Logistic Regression

**What it does.** Fits a straight-line boundary in the feature space. It computes a weighted sum of
the inputs and converts that sum into class probabilities. For three classes it fits three such sums
and picks the largest.

**When it works well.** When classes really are separated by a straight line or plane, when you need
to read the coefficients and say "a one-unit rise in soil moisture reduces the odds of High by this
much", and when you have little data and a complex model would overfit.

**Why it was included.** As a baseline, and specifically to answer the question "does this problem
actually need a non-linear model?" Without it, choosing gradient boosting is an assumption. With it,
the choice is evidence-based.

**How it did here.** Validation Macro F1 0.7796, against roughly 0.9706 for the ensembles. Its
High-class precision was 0.4926, meaning roughly half its High alerts were wrong. That is a decisive
answer: the boundaries in this dataset are not linear.

### XGBoost

**What it does.** Builds decision trees one after another, each trained to correct the errors the
previous ones made. It grows trees level by level and uses both the slope and the curvature of the
loss function to decide where to split and what value each leaf should hold.

**Why it is strong on tabular data.** Real tabular relationships are full of thresholds and
interactions. "Irrigate heavily if moisture is below 20% **and** the crop is flowering **and** the
field is not mulched" is three conditions combined, which is exactly what a path through a decision
tree represents. A linear model would need those interactions hand-built.

**Why it suits agricultural data.** Crop water demand does not respond smoothly to every input.
There are stage thresholds, wilting points and temperature ranges where behaviour changes abruptly.
Trees model breakpoints natively. It also has strong explicit regularisation, which matters when the
minority class has only 14,707 real training examples.

**How it did here.** Validation Macro F1 0.9703, test Macro F1 0.9708 and the best test Macro
ROC-AUC of the four at 0.9976. Statistically level with the other two ensembles.

**Configuration used:** `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`,
`colsample_bytree=0.8`, `min_child_weight=5`, `gamma=0.1`, `reg_lambda=2.0`, `reg_alpha=0.1`.

### LightGBM

**What it does.** The same gradient-boosting idea, with two engineering differences. It buckets
continuous features into histogram bins instead of considering every possible split point, and it
grows trees leaf-wise, always splitting whichever leaf reduces the loss most, rather than filling
each level evenly.

**Why it is efficient.** Histogram binning turns a sort over hundreds of thousands of values into a
pass over a few hundred buckets. On the 558,941 training rows used here, LightGBM trains noticeably
faster than the alternatives.

**When it is useful.** Large datasets, wide feature sets, and any situation where you want to try
many configurations within a time budget. The trade-off is that leaf-wise growth produces deeper,
more irregular trees which can overfit smaller datasets unless the minimum-samples-per-leaf
constraint is set sensibly.

**How it did here.** Validation Macro F1 0.9704 and the best cross-validated Macro F1 of the four at
0.9691. It also had the largest train-to-validation gap (+0.0235), consistent with leaf-wise growth
fitting training data more tightly.

### CatBoost

**What it does.** Gradient boosting again, with two distinctive ideas. It builds **symmetric**
(oblivious) trees, where every node at the same depth uses the same split condition, which acts as a
strong built-in regulariser. And it uses **ordered boosting**, computing the statistics for each row
from rows that came before it in a random permutation, which reduces the subtle self-referential
bias that ordinary boosting introduces.

**Why it is useful for categorical data.** It can consume categorical columns directly, replacing
each category with a target statistic computed in that ordered way. This avoids one-hot encoding
entirely, which matters when a column has many levels.

**Why it fits heterogeneous tabular data.** Datasets like this one mix soil chemistry, weather
readings, crop stage labels and management flags. CatBoost's defaults are conservative and its
symmetric trees resist overfitting on exactly this kind of mixed input.

**How it did here.** Validation Macro F1 0.9706, the highest of the four, and the smallest
train-to-validation gap at +0.0126, which is what the symmetric-tree regularisation predicts.

**The native categorical test.** CatBoost's headline advantage was measured directly: a separate run
gave it the raw categorical columns instead of the one-hot matrix. It scored 0.9701, slightly
*below* the 0.9706 it achieved with the shared encoding. The reason is that the categorical columns
here have between 2 and 6 levels. One-hot encoding a 6-level column costs five extra features, which
is nothing. CatBoost's native handling pays off on columns with hundreds or thousands of levels, and
there are none here.

### How they relate to this dataset specifically

The class-conditional averages explain why the ensembles win so decisively over the linear model:

| Feature | High | Medium | Low |
|---|---:|---:|---:|
| Soil moisture (%) | 17.67 | 29.74 | 43.31 |
| Temperature (C) | 34.57 | 28.89 | 25.35 |
| Wind speed (km/h) | 14.64 | 11.79 | 9.22 |
| Rainfall (mm) | 989.2 | 1444.5 | 1500.5 |
| Humidity (%) | 61.12 | 61.00 | 61.95 |

Soil moisture, temperature and wind separate the classes cleanly and in a consistent direction, but
the boundaries are thresholds rather than a smooth gradient, and they interact with growth stage.
Only 0.16% of fields at sowing are High need, against 6.44% at flowering. Mulching cuts the High
rate from 5.85% to 0.79%. Those are conditional effects, and conditional effects are what tree
ensembles are built for.

### What we learned

Including a linear baseline is not a formality. It converted "we used gradient boosting" from an
assumption into a measured finding, and it is the single easiest question to answer in a viva.

---

## 9. Experimental methodology

The complete path from raw file to served prediction:

```
notebook/train.csv  (630,000 rows)
        |
        |  drop the id column
        v
data-quality audit
   missing values, duplicate rows, contradictory labels, class counts
        |
        v
STRATIFIED SPLIT  (seed 42)
        |
        +---------------------+---------------------+
        |                     |                     |
   TRAINING              VALIDATION              TEST
   441,000 rows           94,500 rows            94,500 rows
        |                     |                     |
        |                     |                  SEALED
        v                     |                     |
fit OneHotEncoder             |                     |
on training rows only         |                     |
        |                     |                     |
        v                     |                     |
apply smote_original_partial
to training rows only         |                     |
        |                     |                     |
        v                     |                     |
carve an inner slice          |                     |
for early stopping            |                     |
        |                     |                     |
        v                     |                     |
train 4 models  ------------->|                     |
        |                     v                     |
        |              compare on validation        |
        |              + 5-fold cross-validation    |
        |                     |                     |
        |                     v                     |
        |              apply the selection rule     |
        |                     |                     |
        |                     v                     |
        |              tune the High threshold      |
        |              (on validation)              |
        |                     |                     |
        |                     v                     |
        +------------> FINAL MODEL: CatBoost <-------+
                              |                     |
                              |                     v
                              |            score ONCE on test
                              |                     |
                              v                     v
                    irrigation_model.pkl      results.json
                    model_schema.json
                              |
                              v
                        Flask /api/predict
                              |
                              v
                     Low / Medium / High
```

Every arrow into the TEST column is one-way and happens once, at the end.

### What we learned

Writing the pipeline as a diagram before writing the code is what caught the ordering mistakes. Each
defect in the earlier version corresponds to an arrow pointing the wrong way in this picture.

---

## 10. Why these evaluation metrics

### Accuracy

**What it is.** The share of predictions that are correct.

**What it is good for.** It is the number everyone understands, and it is the right summary when the
classes are balanced and every mistake costs the same.

**Why it can mislead here.** With 58.7% of rows in the Low class, a model that always predicts Low
scores 58.7% accuracy while being completely useless. Accuracy cannot distinguish that model from a
good one, because the 3.33% of rows it gets catastrophically wrong barely move the average.

**Is it still meaningful?** Yes, in this project, because it is quoted alongside the class-wise
numbers. The final model reaches 98.55% accuracy **and** 0.9219 recall on the rare class, so the
accuracy is not hiding a collapsed minority class. Accuracy on its own would have been misleading;
accuracy plus per-class recall is informative.

### Precision

**What it is.** Of the fields the model flagged as High, what share genuinely were High.

**What it costs when it is low.** False alarms. Water and pump time spent on fields that did not
need it, and, more corrosively, farmers learning to ignore the alerts.

**Here:** 0.9606 on the test split. About 4 in every 100 High alerts is a false alarm.

### Recall

**What it is.** Of the fields that genuinely were High, what share the model caught.

**What it costs when it is low.** Missed fields. A crop that needed water and did not get it, which
at a water-sensitive stage translates directly into lost yield.

**Here:** 0.9219. The model catches 2,905 of the 3,151 genuinely High-need fields in the test split
and misses 246.

### F1

**What it is.** The harmonic mean of precision and recall, which is a single number that is only
high when **both** are high.

**Why the harmonic mean.** It punishes imbalance between the two. A model with 100% recall and 20%
precision, achieved by flagging every field as High, has an arithmetic mean of 60% but an F1 of only
33%. That is the correct verdict on such a model.

**Why it fits irrigation.** Both errors are real and both cost money, so a metric that lets one be
traded away entirely for the other would be the wrong guide.

### Macro F1

**What it is.** Compute F1 separately for each of the three classes, then take the plain average.
Each class contributes equally, regardless of how many rows it has.

**Why it is the right headline metric here.** This is the key point of the whole evaluation design.
Weighted F1 averages the per-class scores in proportion to their support, so the Low class with
58.7% of the rows dominates it. That is why the weighted F1 of 0.9855 sits so much higher than the
Macro F1 of 0.9709: the weighted figure is mostly a report on how well the model handles Low.

Macro F1 gives the 21,009-row High class exactly the same weight as the 369,917-row Low class. A
model that gave up on High could not hide it: its High F1 would be near zero and Macro F1 would fall
by roughly a third. Macro F1 is the metric that cannot be gamed by ignoring the rare class, which is
precisely why it was chosen for model selection.

### ROC-AUC

**What it is.** The probability that the model assigns a higher score to a randomly chosen positive
case than to a randomly chosen negative one. Measured one-vs-rest and averaged across classes.

**What it adds.** It judges the ranking of the probabilities rather than the hard predictions, so it
is independent of where the decision threshold sits. A model can have identical accuracy to another
but better-ordered probabilities, and ROC-AUC is what reveals that.

**A caveat worth stating.** ROC-AUC is optimistic under heavy imbalance, because the false-positive
rate has a very large denominator. All three ensembles here score above 0.997, which is too
compressed to separate them. That is why it was not the selection metric.

**Here:** 0.9973.

### Average precision

**What it is.** The area under the precision-recall curve. It ignores true negatives entirely.

**Why it is included.** It is the imbalance-aware counterpart to ROC-AUC. Because it never rewards
the model for correctly identifying the vast Low class, it gives a more honest picture of
minority-class ranking.

**Here:** macro average precision 0.9870, micro 0.9965. The macro figure is visibly lower than the
micro figure, which is exactly the honest signal: averaging per class rather than pooling exposes
that the High class is harder.

### The metric used for selection

**Macro F1 on the validation split**, with High-class recall as the first tie-break and the smaller
train-to-validation gap as the second. Accuracy, ROC-AUC and average precision are all reported, but
none of them is the deciding number: accuracy because it is dominated by the majority class, ROC-AUC
because it is too compressed to discriminate here, and average precision because it measures ranking
rather than the decisions the system actually makes.

### What we learned

"Why F1 and not accuracy" has a specific answer for this dataset, not a general one. The answer is
that the gap between 0.9855 weighted F1 and 0.9709 Macro F1 is entirely the cost of the rare class,
and only the macro average makes that cost visible.

---

## 11. Model comparison

All numbers below come from one run of the experiment. The validation and cross-validation columns
are what the selection used; the test columns were computed afterwards.

### Validation and cross-validation

| Model | Validation Macro F1 | Cross-validated Macro F1 | Validation accuracy | Validation High recall | Train minus validation Macro F1 |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.9706 | 0.9680 ± 0.0012 | 0.9849 | 0.9191 | +0.0126 |
| LightGBM | 0.9704 | 0.9691 ± 0.0019 | 0.9844 | 0.9184 | +0.0235 |
| XGBoost | 0.9703 | 0.9681 ± 0.0020 | 0.9848 | 0.9156 | +0.0183 |
| Logistic Regression | 0.7796 | 0.7805 ± 0.0045 | 0.8602 | 0.8248 | +0.0763 |

Cross-validation is 5-fold stratified on 150,000 training rows, with the encoder and the resampler
refitted inside every fold so no fold's preprocessing sees its own validation data.

### Test split

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC | Micro AP | High precision | High recall | High F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.9855 | 0.9709 | 0.9855 | 0.9973 | 0.9965 | 0.9606 | 0.9219 | 0.9409 |
| XGBoost | 0.9855 | 0.9708 | 0.9854 | 0.9976 | 0.9970 | 0.9616 | 0.9207 | 0.9407 |
| LightGBM | 0.9851 | 0.9704 | 0.9850 | 0.9976 | 0.9970 | 0.9603 | 0.9210 | 0.9402 |
| Logistic Regression | 0.8631 | 0.7848 | 0.8656 | 0.9618 | 0.9439 | 0.5023 | 0.8334 | 0.6268 |

### Reading this table honestly

The three ensembles differ by less than 0.001 on every metric in both tables. Notice that the two
evaluations disagree about the ranking: the single validation split puts CatBoost first, while
cross-validation puts LightGBM first. That disagreement is the clearest possible evidence that the
difference is noise rather than signal.

Logistic Regression is the one genuinely separated result, and its High-class precision of 0.5023 is
the specific number that rules it out: half of everything it flags as High would be wrong.

### What we learned

Reporting four models that finish within 0.001 of each other is a more useful result than declaring
a winner by a hair. It tells the reader the problem is saturated at this feature set, and that
further gains would have to come from better data rather than a better algorithm.

---

## 12. Why the final model was selected

### The rule, fixed before the comparison

> Rank candidates by validation Macro F1 rounded to four decimals; break ties on validation
> High-class recall and then on the smaller absolute train-validation Macro F1 gap.
> Cross-validation standard deviations are reported alongside so a reader can judge whether the
> winning margin exceeds run-to-run noise.

### Applying it

| Rank | Model | Validation Macro F1 | Validation High recall | Train minus validation gap |
|---:|---|---:|---:|---:|
| 1 | CatBoost | 0.9706 | 0.9191 | +0.0126 |
| 2 | LightGBM | 0.9704 | 0.9184 | +0.0235 |
| 3 | XGBoost | 0.9703 | 0.9156 | +0.0183 |
| 4 | Logistic Regression | 0.7796 | 0.8248 | +0.0763 |

CatBoost ranks first on validation Macro F1 by **0.00014**.

### The honest caveat, stated plainly

That margin is smaller than the seed-to-seed standard deviation measured in section 6 (0.0004) and
smaller than every cross-validation standard deviation in section 11. CatBoost, LightGBM and XGBoost
are **statistically indistinguishable** on this dataset. If a panel asks "is CatBoost really better
than XGBoost here?", the correct answer is no, and the evidence for that is in this document.

### So why CatBoost is the one deployed

Three reasons, in order of weight:

1. **The rule was declared before the comparison and CatBoost came first under it.** Changing the
   rule after seeing the results, in order to keep the previously deployed model, would be exactly
   the post-hoc reasoning that makes a result untrustworthy.
2. **It has the smallest train-to-validation gap** at +0.0126, against +0.0183 for XGBoost and
   +0.0235 for LightGBM. When scores are tied, the model that leans least on its training data is
   the safer one to put in front of real fields. This is also the theoretically expected consequence
   of CatBoost's symmetric trees and ordered boosting.
3. **Its cross-validation spread is the tightest** at ±0.0012, against ±0.0020 and ±0.0019. It is
   the most consistent of the three across resamples of the training data.

### What changed from the earlier version, and why

The earlier version deployed XGBoost, justified by a Macro ROC-AUC of 0.9976 against LightGBM's
0.9975: a margin of 0.0001, which is noise. Under the corrected methodology XGBoost is still an
excellent model, with the best test Macro ROC-AUC of the four at 0.9976, but it does not come first
under a rule that was fixed in advance and applied to validation data.

The model was changed because the evidence pointed elsewhere, not because CatBoost is better in any
meaningful sense. Had the rule ranked XGBoost first, XGBoost would have stayed.

### What we learned

Declaring the selection rule before looking at the results is what makes a close outcome defensible.
Without a pre-declared rule, any of the three could have been justified after the fact, and none of
those justifications would have been worth anything.

---

## 13. Overfitting analysis

The panel's question about the earlier version was whether metrics near 98% indicate a model that
memorised its training data. This section answers it with evidence rather than assertion.

### The three scores

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Training | 0.9843 | 0.9831 |
| Validation | 0.9849 | 0.9706 |
| Test | 0.9855 | 0.9709 |

### The gaps

| Gap | Accuracy | Macro F1 |
|---|---:|---:|
| Train minus validation | -0.0005 | +0.0126 |
| Train minus test | -0.0012 | +0.0123 |
| Validation minus test | -0.0006 | -0.0003 |

### What these say

**Test accuracy is higher than training accuracy**, by 0.0012. An overfitted model shows the
opposite: near-perfect training scores and a visible drop on unseen data. This model does not.

**Validation and test agree to 0.0003 Macro F1.** Two independent samples of unseen data produce the
same answer, which is what a stable model looks like.

**The Macro F1 training gap of +0.0123 is not overfitting**, and this distinction matters. Training
Macro F1 is computed on the resampled training set, where the High class makes up about 18% of rows
instead of its natural 3.3%. Macro F1 averages per-class scores, and the High class is easier to
score well on when it is abundant and partly synthetic. Accuracy, which is computed identically on
both, shows no gap whatsoever. The gap is an artefact of the resampled training distribution, not
memorisation.

**Cross-validation confirms it.** Across 5 folds, CatBoost scored 0.9680 ± 0.0012. A standard
deviation of 0.0012 means the result does not depend on which rows happened to land in which split.

### Controls actually applied

- Early stopping against an inner slice of the training data, keeping validation free for selection
- Learning rate 0.05 (shrinkage: each tree contributes only a small correction)
- Tree depth capped at 6
- L2 leaf regularisation of 5
- Symmetric trees, which constrain every node at a given depth to share a split condition
- 5-fold stratified cross-validation to quantify fold-to-fold variance

One honest note on the training budget: the boosting round cap of 800 was reached before early
stopping triggered for all three ensembles. The models were still improving very slowly at that
point. The cap was a deliberate compute limit rather than a tuned value, and since the
generalisation gaps stayed small throughout, stopping there was reasonable. A longer budget would
likely have moved the third decimal place and nothing more.

### So why are the scores so high?

Because the dataset is easy, and this is the honest answer to give a panel.

The evidence: a completely plain pipeline with no resampling, default-ish hyperparameters and no
tuning already reaches 0.9703 Macro F1. Class membership is driven almost entirely by a handful of
columns with unusually clean separation. Mean soil moisture is 17.7% for High against 43.3% for Low.
Mean temperature is 34.6 against 25.3 degrees Celsius. Only 0.16% of sowing-stage fields are High
need against 6.44% at flowering. Real field data is never this tidy.

The combination of 630,000 rows, zero missing values, zero duplicates, zero contradictory labels and
clean threshold separation is consistent with labels generated from a rule over these features. The
high scores measure how well a model can recover that rule, which it can do very well. They do not
measure how well it would predict irrigation need on a real farm. This is recorded as the project's
principal limitation in section 20, not hidden.

### What we learned

"The test score is high, therefore the model is good" is not an argument. "Test accuracy exceeds
training accuracy, validation and test agree to four decimal places, cross-validation varies by
0.0012, and an untuned baseline already scores nearly the same" is an argument, and it supports a
narrower conclusion: the model is not overfitted, and the dataset is easy.

---

## 14. Final test results

Computed once, after every decision was fixed, on 94,500 rows the model had never seen.

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

These are the only authoritative numbers for this project. The application, the README and this
document all read them from `experiments/results.json`.

---

## 15. Class-wise results

| Class | Precision | Recall | F1 | Test rows |
|---|---:|---:|---:|---:|
| High | 0.9606 | 0.9219 | 0.9409 | 3,151 |
| Medium | 0.9856 | 0.9760 | 0.9808 | 35,861 |
| Low | 0.9868 | 0.9952 | 0.9910 | 55,488 |

### What each row means in the field

**High** (0.9409 F1). Precision 0.9606: about 4 in every 100 High alerts is a false alarm, costing a
wasted irrigation cycle. Recall 0.9219: the model finds 2,905 of the 3,151 genuinely High-need
fields and misses 246. This is the weakest class, which is expected: it has the fewest real training
examples.

**Medium** (0.9808 F1). The middle class is always the hardest boundary in an ordinal problem
because it borders both neighbours. Recall 0.9760 is the lowest of the three, and section 16 shows
where those errors go.

**Low** (0.9910 F1). Nearly perfect, which is unsurprising: it is the largest class with the most
training examples and the most distinctive profile of high soil moisture and low temperature.

### Why per-class recall matters most for High

An irrigation system that misses a water-stressed field costs yield, and at flowering that loss can
be permanent for the season. An irrigation system that raises a false alarm costs one unnecessary
watering. The two errors are not symmetric in consequence, which is why High recall is quoted
prominently and used as the first tie-break in model selection, and why section 11 of the README
documents how the decision threshold could be shifted if a particular farm wanted to weight them
differently.

---

## 16. Confusion matrix analysis

Rows are what the field actually needed, columns are what the model predicted.

| | Predicted High | Predicted Low | Predicted Medium | Row total |
|---|---:|---:|---:|---:|
| **Actually High** | 2,905 | 0 | 246 | 3,151 |
| **Actually Low** | 0 | 55,223 | 265 | 55,488 |
| **Actually Medium** | 119 | 740 | 35,002 | 35,861 |

### Reading it as irrigation decisions

**2,905 High fields correctly identified.** These get priority water and the crop is protected.

**246 High fields predicted as Medium.** These get a normal dose instead of a priority dose:
under-watered, but not ignored. In an ordinal problem this is a one-step error, and the practical
cost is partial rather than total.

**0 High fields predicted as Low.** Zero. This is the most important cell in the entire matrix and
the model never lands in it. The catastrophic failure mode, where a field in water stress is marked
as needing nothing, does not occur once in 94,500 test rows. It is worth being precise about what
this does and does not prove: it holds on this test split with this model, and the ordinal structure
of the problem makes it unsurprising, but it is not a guarantee.

**740 Medium fields predicted as Low.** The largest single error cell and the model's real practical
weakness. These fields are under-watered. The consequence is milder than a missed High field, but it
is 740 fields.

**119 Medium fields predicted as High.** Over-watered. Wasteful, but agronomically safe.

**265 Low fields predicted as Medium, and 0 predicted as High.** Water spent on fields that did not
need it, at a rate of about 0.5% of Low fields.

### The pattern

Every error is a one-step error along the Low - Medium - High ordering. Not a single prediction
jumps from one end of the scale to the other. That is the behaviour you want from an ordinal
classifier: when it is wrong, it is wrong by a little.

### The High-class trade-off, quantified

From the threshold study on the validation split, lowering the probability required to call a field
High from 0.5 to 0.3 raises High recall from 0.9191 to 0.9308 while dropping High precision from
0.9647 to 0.9483. A farm that considers a missed field much more costly than a wasted watering could
take that trade. The default is kept because it maximises Macro F1, and because the choice belongs
to the operator rather than to the model author.

### What we learned

The confusion matrix says more about whether this system is safe to deploy than any single metric
does. 0.9709 Macro F1 does not tell you that no High field is ever called Low. The matrix does.

---

## 17. Deployment

### The artifact

`irrigation_model.pkl` is a single scikit-learn `Pipeline` containing two things: the
`ColumnTransformer` holding the `OneHotEncoder` that was fitted on the training rows, and the
trained CatBoost classifier. They are saved together, not separately.

This is the central design decision of the deployment. The application never encodes anything
itself. It hands the pipeline a row of raw field values, and the pipeline applies exactly the
encoding it was fitted with during training.

`model_schema.json` records what a valid request looks like: the raw column order, the allowed
levels for each categorical column, the minimum, quartiles, median and maximum of each numeric
column, and the mapping from class index to label ({"0": "High", "1": "Low", "2": "Medium"}).

`experiments/results.json` holds the complete experiment record and is the only source of the
metrics the web pages display.

### Flask

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | prediction dashboard |
| `/models` | GET | model specification and the full experiment record |
| `/history` | GET | session prediction log |
| `/api/predict` | POST | prediction from raw field values |
| `/api/models` | GET | metrics for all four candidates plus the final model |
| `/api/schema` | GET | the feature schema |
| `/api/history` | GET | recent predictions |

`/api/predict` accepts either the exact training column names (`Soil_Moisture`) or lower-case
aliases (`soil_moisture`, `windspeed_kmph`) so earlier clients keep working. It validates every
categorical value against the schema and returns the list of allowed levels when one is wrong,
rather than scoring a field the model has never seen.

### Real-time prediction path

```
POST /api/predict with 19 raw field values
        |
        v
map aliases to the training column names
        |
        v
validate categories against model_schema.json
   unknown value -> 400 with the allowed levels
        |
        v
build a one-row DataFrame in the schema's column order
        |
        v
irrigation_model.pkl  (fitted encoder -> CatBoost)
        |
        v
class label + probabilities for all three classes
        |
        v
reasoning generated by comparing the inputs against the
recorded training quartiles, plus feature importances
```

### Render

The application runs on Render behind gunicorn (`Procfile`: `gunicorn app:app`). `requirements.txt`
carries only what serving needs: Flask, gunicorn, joblib, numpy, pandas, scikit-learn and CatBoost.
The experiment dependencies (XGBoost, LightGBM, imbalanced-learn, matplotlib, Jupyter) live in
`requirements-dev.txt` so deployments stay small.

The model is loaded once at start-up, not per request.

### What we learned

Serialising the encoder and the model together, instead of saving a column list and rebuilding the
encoding by hand in the web application, is what turns train-serving consistency from a hope into a
property of the artifact.

---

## 18. Train-serving consistency

### The problem, in general

Train-serving skew is when the features a model receives in production differ from the features it
was trained on. It is insidious because nothing crashes. The model returns a confident answer
computed from inputs that mean something different from what it learned.

### The problem, in this project specifically

The earlier deployment had it, badly, and it had gone unnoticed:

| What the web form offered | What the training data contains |
|---|---|
| Crop: Rice, Wheat, Maize, Cotton, Sugarcane, **Soybean, Tomato, Potato, Onion, Sunflower** | Cotton, Maize, Potato, Rice, Sugarcane, Wheat |
| Stage: **Germination, Seedling**, Vegetative, Flowering, **Fruit Development, Maturity**, Harvest | Flowering, Harvest, Sowing, Vegetative |
| Water source: **Canal, Borewell, Drip System, Sprinkler** | Groundwater, Rainwater, Reservoir, River |
| "Recent Rainfall", default 2.0 mm | `Rainfall_mm` from 0 to 2500, median 1467 |
| Soil moisture slider 0 to 100 | 8.0 to 65.0 |
| Soil type, season, irrigation method, region: not asked, hardcoded | four real features the model uses |

Every bolded value produced an all-zero one-hot block, so the model silently scored the field as the
dropped reference category. All four water-source options were invalid, which meant that feature was
effectively constant in production. `Rainfall_mm` was being fed values around 2 when the model had
only ever seen values in the hundreds and thousands. Four features the model relies on were frozen
at defaults the user could not change.

### How it was fixed

**One artifact, one encoding.** The encoder and classifier are one pipeline, so there is no second
encoding step in the application that could drift.

**The form is generated from the schema.** Every dropdown in `templates/index.html` is rendered by a
Jinja macro that loops over `schema.categorical_levels[column]`, and every numeric input takes its
minimum, maximum and default from `schema.numeric_ranges[column]`. It is not possible to offer an
option the model has not seen, because the options are read from the training data.

**All 19 features are collected.** Soil type, season, irrigation method and region are now form
fields instead of hardcoded constants.

**Invalid input is rejected, not guessed.** A request containing `Crop_Type: Tomato` returns HTTP
400 with the six crop types the model actually knows, instead of silently scoring the field as
Cotton.

**Ranges match the training distribution.** `Rainfall_mm` now defaults to the training median of
1467.16 with the true minimum and maximum as bounds, and the label says "Cumulative Rainfall" rather
than "Recent Rainfall".

### How the fix is verified

`experiments/verify_deployment.py` is a check that can be run against any deployment:

1. Rebuild the same split with the same seed.
2. Score the untouched test rows through the saved artifact.
3. Assert that accuracy, Macro F1 and the full confusion matrix match `experiments/results.json`.
4. Send raw rows through the live HTTP API and assert the labels and probabilities match the offline
   pipeline exactly.

Current output:

```
Feature order matches schema: True
PASS  test accuracy: 0.985503 (recorded 0.985503)
PASS  test macro F1: 0.970893 (recorded 0.970893)
PASS  test confusion matrix reproduced from the artifact
PASS  25 raw rows served over HTTP match the offline pipeline
```

### What we learned

This was the most serious defect found in the entire audit, and it was in the part of the project
nobody was questioning. The panel asked about overfitting, which turned out to be fine. The actual
problem was that live predictions were being computed from inputs the model had never seen.

---

## 19. Frontend

### Guided input

A four-step form: crop and field, weather, soil and water, then review. Splitting nineteen inputs
across four steps keeps each screen readable and lets each step be validated before moving on.

Every dropdown is generated from `model_schema.json`, so the options are exactly the categories
present in the training data. Every numeric input carries the training minimum, maximum and median,
and shows the training range as helper text, so a user can see what the model considers a normal
value.

### Presets

Three presets fill the form from the recorded training quartiles: a hot dry field (soil moisture and
rainfall at the 25th percentile, temperature and wind at the 75th), a wet field after rain (the
reverse), and a median field. They are named for the conditions they set, not for the class they
produce, because the model decides the class.

### Prediction result

- The predicted class as a colour-coded banner
- Confidence, which is the highest class probability
- A probability bar for each of Low, Medium and High
- Reasoning generated by comparing each input against the recorded training quartiles, for example
  "Soil moisture of 15 percent sits in the driest quarter of the training data"
- The top feature importances from the deployed model

Every one of these comes from the `/api/predict` response. Nothing on the result card is generated
in the browser, so what the user sees is what the model actually computed.

### Model specification page

`/models` renders entirely from `experiments/results.json`: the final test metrics, the class-wise
report, the four-model comparison with validation, cross-validation and test columns, the
overfitting table with all three gaps, the test confusion matrix, and the experimental protocol. It
also carries a short section stating that the winning margin is inside the noise. There are no
hardcoded numbers in the templates.

### History

`/history` lists the predictions made during the current server session, with the model name
attached to each. The log is in memory and resets when the server restarts, which is deliberate:
this is a demonstration application, not a system of record.

### What we learned

Once the frontend reads its numbers from the experiment record, the class of bug where documentation
and deployment disagree disappears by construction. There is no second copy of the metrics to fall
out of date.

---

## 20. Limitations

Stated plainly, because a panel will find these anyway and it is better to have named them first.

### The dataset appears to be synthetic

This is the most important limitation. The evidence: 630,000 rows with zero missing values, zero
duplicate rows, zero contradictory labels, and class separation far cleaner than field data ever is
(mean soil moisture 17.7% for High against 43.3% for Low). An untuned pipeline with no resampling
already reaches 0.9703 Macro F1.

The most likely explanation is that the labels were generated by a rule over these features. If so,
98.55% accuracy measures how well the model recovered that rule, not how well it would predict
irrigation need on a real farm. Every number in this document is a correct measurement of
performance on this dataset, and none of them is a promise about a real field.

### Synthetic training rows

85,293 of the High training rows are SMOTE interpolations rather than observed fields, and they
carry fractional values in the one-hot columns that no real field could have. They were kept because
they measurably improved validation Macro F1, and they are confined to the training split, but they
are an assumption about what lies between observed fields rather than evidence.

### No temporal structure

Rows are independent. A real irrigation decision depends on what happened over the previous days:
whether it rained yesterday, how fast the soil is drying, when the field was last watered.
`Previous_Irrigation_mm` is a single scalar, not a history. The model cannot see a trend.

### No live sensors

Every input is typed in or supplied by the caller. A deployed system would read soil moisture probes
and a weather feed. The gap between a hand-entered moisture estimate and a calibrated probe reading
is real and unquantified here.

### Geography is a five-level label

`Region` takes one of Central, East, North, South, West. There is no soil map, no elevation, no
reference evapotranspiration, no groundwater depth. Two fields in the same region can have entirely
different water requirements.

### The model outputs a class, not a volume

The dataset contains no water-volume target, so the system cannot recommend millimetres. The API
returns `water_recommendation_mm: null` and states `water_recommendation_available: false` rather
than inventing a number.

### Distribution shift is unhandled

The model assumes future fields resemble the training distribution. Climate trends, a new crop
variety, a different soil type or an unusual season would all place inputs outside what it has seen.
There is no drift monitoring and no retraining trigger.

### The choice of final model is not strongly supported

CatBoost, LightGBM and XGBoost differ by less than the measurement noise. The selection rule
produces a defensible answer, but it is a decision made under genuine uncertainty rather than a
demonstration that CatBoost is superior.

### The boosting budget was a compute limit

All three ensembles reached the 800-round cap before early stopping triggered. They were still
improving marginally. The cap was chosen for tractability, not tuned.

---

## 21. Future improvements

**Validate on real field data.** The single highest-value next step. Even a few hundred labelled
observations from an actual farm would establish whether the model transfers or whether it has only
learned a generator's rule.

**Add temporal features.** Days since last irrigation, a soil-moisture trend over the previous week,
cumulative rainfall over trailing windows, and a running growing-degree-day total. This would change
the problem from a snapshot classification to a time-series one, which is closer to how the decision
is actually made.

**Predict a volume, not a class.** With a water-applied target, the same features could support a
regression that outputs millimetres. That is the output a farmer can act on directly.

**Reference evapotranspiration.** The FAO Penman-Monteith equation combines temperature, humidity,
wind and radiation into a physically grounded water-demand estimate. Adding it as an engineered
feature would introduce agronomic knowledge the model currently has to infer.

**Cost-sensitive thresholds.** Section 16 shows the precision-recall trade-off for the High class.
Given the cost of a missed field and the cost of a wasted watering for a specific farm, the
operating point could be chosen to minimise expected cost rather than to maximise Macro F1.

**Calibration analysis.** The interface presents the highest class probability as a confidence.
Whether a stated 90% confidence corresponds to being right 90% of the time has not been measured. A
reliability diagram and, if needed, isotonic or Platt calibration would make that claim defensible.

**Per-field explanations with SHAP.** The current reasoning compares inputs against training
quartiles, which is honest but generic. SHAP values would attribute a specific contribution to each
feature for each individual prediction.

**Drift monitoring.** Track the distribution of incoming requests against the training ranges
recorded in `model_schema.json` and raise a warning when live inputs move away from what the model
was trained on.

**Sensor integration.** Read soil moisture from IoT probes and weather from an API, so the system
runs continuously rather than waiting for a form submission.

---

## 22. Viva and panel defence

### 1. Why is the dataset imbalanced?

Because irrigation need is genuinely a rare event. Most fields on most days are adequately watered,
so the Low class dominates at 58.72% while High, the crisis state, is only 3.33%. The ratio is 17.61
to 1. It is a property of the phenomenon, not a collection error, which is why the test split
deliberately preserves the natural proportions.

### 2. How did you detect the imbalance?

By counting the target values, computing the majority-to-minority ratio, and then recomputing the
same counts inside each of the three splits to confirm stratification preserved them. All of it is
recorded in `experiments/results.json` under `dataset.class_distribution` and `split.class_counts`.

### 3. Why does class imbalance matter?

Training minimises average error, so a rare class contributes little to the loss and the model can
improve its score by ignoring it. Here a model that always predicted Low would score 58.7% accuracy
while missing every single one of the 3,151 High-need fields in the test split. The class it would
ignore is the one that matters most.

### 4. What is SMOTE?

Synthetic Minority Over-sampling Technique. For each minority-class row it finds the five nearest
minority neighbours, picks one at random, and creates a new point somewhere on the straight line
between them. Unlike simple duplication it produces points the model has not seen before, which
gives the decision boundary more to work with.

### 5. Why did you use SMOTE?

Because it measurably won a controlled comparison, not because it is conventional. Eight strategies
were tested under one probe model on one split. Partial SMOTE scored 0.9709 validation Macro F1,
against 0.9703 for no resampling and 0.9636 for SMOTENC. Five-seed repetition put the noise floor at
0.0004, and only partial SMOTE cleared one standard deviation below the best. It is a small margin,
and no resampling would have been an acceptable alternative.

### 6. Why should SMOTE only be applied to training data?

A synthetic point sits between two real training points, so the model has effectively already seen
it. If such points reached the validation or test split, the model would be scored partly on data
derived from what it learned, and the score would be inflated. Here SMOTE runs after the split, on
the 441,000 training rows only. Validation and test keep the natural class proportions (3,151 High
rows in test, matching the population rate).

### 7. What is data leakage?

Any situation where information that would not be available at prediction time influences training
or evaluation. It has three common forms: a feature derived from the target, preprocessing fitted on
data that includes the test set, and the test set being consulted during tuning or selection. The
result is always the same: a score that looks good in the notebook and does not survive contact with
real data.

### 8. How did you prevent leakage?

Six specific measures:

1. The split happens on raw rows, before any encoding or resampling.
2. The `OneHotEncoder` is fitted on training rows only.
3. Resampling runs on training rows only.
4. Early stopping uses an inner slice of the training split, not the validation split.
5. Model selection uses validation only; the test split is scored once at the end.
6. The High-class threshold study runs on validation, not test.

Zero index overlap between splits is asserted in code. Point 6 is the direct correction of a defect
in the earlier version, where that sweep was run on the test set.

### 9. Why did you report accuracy?

Because it is the number everyone understands and, in this project, it is not misleading: it is
quoted alongside per-class recall. The final model reaches 98.55% accuracy **and** 0.9219 recall on
the rare class, so the headline figure is not concealing a collapsed minority class. Reported alone
it would have been misleading; reported with the class-wise table it is informative.

### 10. Why did you choose Macro F1 for selection?

Because it is the metric that cannot be gamed by ignoring the rare class. Macro F1 computes F1 per
class and averages them equally, so the 21,009-row High class counts as much as the 369,917-row Low
class. The evidence that this matters is in the numbers: weighted F1 is 0.9855 while Macro F1 is
0.9709. That gap is entirely the cost of the rare class, and only the macro average makes it
visible.

### 11. Why is recall important for High irrigation demand?

Because the two errors are not symmetric. A false alarm costs one unnecessary watering. A missed
High-need field costs yield, and at flowering that loss is permanent for the season. High recall is
0.9219: the model catches 2,905 of 3,151 genuinely High-need fields. It is also the first tie-break
in the selection rule, and section 16 documents how the threshold could be lowered to trade
precision for more recall if a particular farm wanted that.

### 12. Why not XGBoost, which was your previous model?

XGBoost is still excellent here. It has the best test Macro ROC-AUC of the four at 0.9976 and a test
Macro F1 of 0.9708, which is 0.0001 from the deployed model. It simply did not come first under a
selection rule that was fixed before the comparison and applied to validation data: 0.9703 against
0.9706. The earlier justification for XGBoost was a 0.0001 ROC-AUC margin over LightGBM, which is
noise. The model changed because the rule pointed elsewhere, not because XGBoost is inadequate.

### 13. Why CatBoost, then?

Three reasons in order of weight. First, the pre-declared rule ranks it first on validation Macro F1
(0.9706). Second, it has the smallest train-to-validation gap at +0.0126, which is the theoretically
expected consequence of its symmetric trees and ordered boosting, and when scores are tied the model
that leans least on its training data is the safer deployment. Third, it has the tightest
cross-validation spread at ±0.0012.

I would add immediately that the margin over the runner-up is 0.00014, which is inside the noise.
The three ensembles are statistically indistinguishable and I would not claim CatBoost is better in
any meaningful sense.

### 14. Why not LightGBM?

It is the closest competitor and arguably has a claim: cross-validation actually ranks it highest at
0.9691 against 0.9680. It lost on the single validation split by 0.00014, and it has the largest
train-to-validation gap of the three at +0.0235, which is consistent with leaf-wise tree growth
fitting the training data more tightly. Deploying LightGBM would have been defensible; the rule was
declared first and it did not come first under it.

### 15. Why include Logistic Regression at all if it lost so badly?

Precisely because it lost badly. Without a linear baseline, choosing gradient boosting is an
assumption. With one, it is a measurement. Logistic Regression reached 0.7796 Macro F1 and, more
tellingly, 0.5023 precision on the High class, meaning about half its High alerts would be wrong.
That is direct evidence that the class boundaries are non-linear and that the extra complexity of an
ensemble is earning its place.

### 16. Which model is best for tabular data?

Gradient-boosted tree ensembles, as a family, and this experiment is consistent with that: all three
scored around 0.9709 Macro F1 while the linear model reached 0.7796. Tabular data is full of
thresholds and interactions between heterogeneous columns, which is what trees represent natively.
Within the family the choice is usually a matter of engineering preference rather than accuracy,
which is exactly what the 0.00014 spread here shows.

### 17. What does each boosting library do differently?

XGBoost grows trees level by level and uses second-order gradient information with explicit L1 and
L2 penalties. LightGBM buckets features into histograms and grows leaf-wise, always splitting the
leaf that reduces loss most, which makes it fastest. CatBoost builds symmetric trees where every
node at a given depth shares a split condition, and uses ordered boosting to reduce the
self-referential bias in target statistics. The symmetric constraint is a strong regulariser, which
shows up here as the smallest train-to-validation gap.

### 18. CatBoost is meant to be good at categorical features. Did that help?

It was measured directly, and no. A separate run gave CatBoost the raw categorical columns instead
of the one-hot matrix, and it scored 0.9701 against 0.9706 with the shared encoding. The reason is
that the categorical columns here have between 2 and 6 levels. One-hot encoding is cheap at that
size. Native handling pays off on columns with hundreds or thousands of levels, and there are none
here. So CatBoost is deployed for its regularisation behaviour, not for the feature it is best known
for.

### 19. How did you check for overfitting?

By comparing training, validation and test performance and looking at the gaps. Test accuracy
(0.9855) is actually **higher** than training accuracy (0.9843), and validation and test agree to
0.0003 Macro F1. Cross-validation across 5 folds gave 0.9680 ± 0.0012. An overfitted model shows
near-perfect training scores and a clear drop on unseen data; this shows no drop at all.

The one gap that exists, +0.0123 in Macro F1, has a specific explanation: training Macro F1 is
measured on the resampled training set where High is about 18% of rows instead of 3.3%, and Macro F1
is sensitive to that change of mix. Accuracy, measured identically on both, shows no gap.

### 20. But 98.55% accuracy still looks too good. Why should we believe it?

You are right to press on it, and the honest answer is that the model is not exceptional, the
dataset is easy. A completely plain pipeline with no resampling and no tuning already reaches 0.9703
Macro F1. The dataset has 630,000 rows, zero missing values, zero duplicates, zero contradictory
labels, and class separation far cleaner than field data ever is: mean soil moisture 17.7% for High
against 43.3% for Low.

That pattern is consistent with labels generated from a rule over these features. If so, the score
measures how well the model recovered that rule. It is a correct measurement of performance on this
dataset and it is not a claim about real farms. That is recorded as the project's principal
limitation.

### 21. How can you prove the test results are reliable?

Three ways. First, the process: the test split was separated before any preprocessing, never used to
fit an encoder, generate synthetic rows, stop training, tune a threshold or choose a model, and
scored exactly once. Zero index overlap between splits is asserted in code. Second, the agreement:
validation and test differ by 0.0003 Macro F1, and cross-validation across 5 independent folds
varies by only 0.0012. Third, reproducibility: seed 42 is fixed throughout, and
`experiments/verify_deployment.py` rebuilds the split and reproduces the accuracy, Macro F1 and the
entire confusion matrix from the saved artifact.

### 22. What happens if real-world data differs from the training distribution?

Performance would degrade, and the honest position is that the degree is unknown because there is no
real-world validation set. Three specific exposures: the dataset appears synthetic, so real fields
may follow different relationships; there is no drift monitoring, so a shift would go undetected;
and the model has never seen inputs outside the recorded training ranges.

The system does make the exposure visible rather than hiding it. `model_schema.json` records the
training range of every column, the web form displays those ranges, and the API rejects categorical
values it has never seen rather than silently scoring them as a reference category. The obvious next
step is to log incoming requests against those ranges and warn when live inputs drift away.

### 23. What are the limitations of your approach?

The dataset appears synthetic, so the scores may not transfer. There is no temporal structure, so
the model cannot see how fast a field is drying. There is no live sensor input. Region is a
five-level label with no soil map or evapotranspiration data behind it. The output is a class rather
than a water volume, because the data contains no volume target. Roughly 85,293 High training rows
are synthetic interpolations. And the choice between the three ensembles rests on a margin smaller
than the measurement noise.

### 24. How would you improve this as research work?

The highest-value step is validating against real field data, even a few hundred labelled
observations, because that is what would establish whether any of this transfers. After that: add
temporal features so the model sees drying trends rather than snapshots; move from class prediction
to predicting an actual water volume; add FAO Penman-Monteith reference evapotranspiration as a
physically grounded feature; choose the decision threshold by expected cost rather than by Macro F1;
and check whether the stated confidences are calibrated, since the interface presents them as if
they were.

### 25. If the three models are equivalent, why not just deploy the simplest one?

That would also have been defensible, and it is worth saying so. The counter-argument is that all
three are equally simple to deploy: they are the same size of artifact, the same inference cost, the
same one-line prediction call. There is no simplicity to be gained by choosing differently. Where a
genuine simplicity argument did apply was the imbalance strategy, and there the rule explicitly
favoured the least-assuming option among those that were statistically tied.

### 26. Why not tune hyperparameters more aggressively?

Because the gains available are smaller than the noise. The three ensembles already sit within
0.00014 of each other, seed-to-seed variation is 0.0004, and cross-validation varies by up to
0.0020. Any improvement from extensive tuning would be indistinguishable from chance, and each
additional configuration evaluated on validation increases the risk of selecting a model that fits
that particular split. A stable, defensible model was preferred over a marginally higher score.

---

## 23. Reproducing this

```bash
pip install -r requirements-dev.txt
python3 experiments/run_experiment.py     # roughly 40 minutes
python3 app.py
python3 experiments/verify_deployment.py  # in another shell
```

Seed 42 fixes the split, the resampler and every model.

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

`experiments/run_experiment.py` and `notebook/train_model.ipynb` execute the same code; the notebook
cells are generated from the script so the two cannot disagree.

---

## 24. Summary

A multi-class irrigation-need classifier reaching 98.55% accuracy and 0.9709 Macro F1 on a genuinely
untouched test split, with 0.9219 recall on the rare High class and no High-need field ever
predicted as Low.

The methodology was rebuilt rather than patched. Encoding is fitted after the split, resampling is
confined to the training rows, early stopping runs against an inner training slice so validation
stays clean, threshold analysis moved from test to validation, the imbalance strategy was chosen by
measuring eight options against a noise floor, and the final model was chosen by a rule fixed before
the comparison.

Two findings are worth carrying into the viva. The high scores are explained by an easy, probably
synthetic dataset rather than by an exceptional model, and saying so is more credible than defending
them. And the most serious defect found was not in the modelling at all: the deployed web form was
offering crop types, growth stages and water sources that did not exist in the training data, so
live predictions were being computed from inputs the model had never seen. That is fixed, and the
fix is verified by a script rather than asserted.
