"""
train_godclass.py - God Class Detection using Random Forest
XDD Research - Stage 4: Machine Learning

========================================================
METHODOLOGY ALIGNMENT (thesis s3.2 - s3.5)
========================================================

  STEP 1  - Data loading, cleaning, label-source recording      s3.2
  STEP 2  - EDA + Mann-Whitney U + effect sizes + correction    s3.3
             (rank-biserial r, Cliff's delta, Bonferroni alpha)
  STEP 3  - Stratified instance-level split (80/20)             s3.4
  STEP 4  - Pipeline: StandardScaler -> SMOTE -> RF + OOB        s3.3, s3.4
  STEP 5  - GridSearchCV hyperparameter tuning (macro-F1)       s3.4
             Grid: n_estimatorsin{100,200,500},
                   max_depthin{None,10,20},
                   min_samples_leafin{1,2,5},
                   class_weightin{balanced,None}
  STEP 6  - Learning curve analysis                             s3.4, s5.5
  STEP 7  - Probability calibration (Isotonic + reliability)    s3.5
  STEP 8  - 10-fold stratified CV: F1, Prec, Rec, AUC, MCC,    s3.5
             PR-AUC
  STEP 9  - Final test-set evaluation + bootstrap 95% CIs       s3.5
  STEP 10 - McNemar's test vs PHPMD + Cohen's g + FP/FN         s3.5
  STEP 11 - Feature importance: MDI + permutation cross-check   s3.4, s4.5
  STEP 12 - SHAP TreeExplainer (all 4 outputs + interactions)   s3.4, s4.5
  STEP 13 - Sensitivity analysis: label source                  s3.2.2
  STEP 14 - Per-project generalization analysis                 s3.5, s4.7
  STEP 15 - Model persistence + reproducibility manifest
  STEP 16 - Final summary

========================================================
DATASET OBSERVATIONS (from pre-analysis)
========================================================
- 19,419 total classes across 22 PHP projects (CodeIgniter dropped)
- ~1,820 God Classes (9.4%) - class imbalance handled via SMOTE
- ATFD = 0 for ALL classes - excluded (PhpMetrics 2.9.x limitation)
- NOA = 0 for all  - excluded (PhpMetrics 2.9.x limitation)
- DIT only has values 0 and 1 - low variance, retained (s3.3)
- NOC is 92% zero  - sparse but retained (captures hierarchy roots)
- CBO max=843, WMC max=568 - extreme outliers; StandardScaler used
========================================================
FEATURE SET  (9 features, s3.3)
========================================================
  WMC   Weighted Method Count        - complexity signal
  LCOM  Lack of Cohesion of Methods  - cohesion signal
  CBO   Coupling Between Objects     - coupling signal
  RFC   Response For a Class         - interaction breadth
  DIT   Depth of Inheritance Tree    - inheritance depth
  NOC   Number of Children           - hierarchy root signal
  TCC   Tight Class Cohesion         - cohesion complement
  NOM   Number of Methods            - size signal
  LOC   Lines of Code                - size signal
"""

# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================

import sys
import warnings
import os
import json
import time
import platform
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Preprocessing - s3.3: Z-score normalization = StandardScaler
from sklearn.preprocessing import StandardScaler, label_binarize

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Cross-validation & tuning
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    learning_curve,
)

# Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,         # calibration quality
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)

# Statistical tests
from statsmodels.stats.contingency_tables import mcnemar

# Permutation importance
from sklearn.inspection import permutation_importance

# Model persistence [E10]
import joblib

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed - SHAP outputs (Step 13) will be skipped.")
    print("         Install with: pip install shap\n")

# Visualisation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RUN_START = time.time()

# ==============================================================================
# WINDOWS UTF-8 FIX
# ==============================================================================
# Prevents UnicodeEncodeError on Windows terminals when printing
# Greek letters, arrows, or scientific notation symbols.
# ==============================================================================

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Logging setup
logging.basicConfig(
    filename="experiment.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Experiment started")

print("All imports successful.\n")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_PATH   = SCRIPT_DIR / "output" / "ck_metrics_all.csv"
OUTPUT_DIR  = SCRIPT_DIR / "output" / "ml"
SHAP_DIR    = OUTPUT_DIR / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Feature set - s3.3 (9 OO metrics after exclusions)
FEATURES = ["WMC", "LCOM", "CBO", "RFC", "DIT", "NOC", "TCC", "NOM", "LOC"]
TARGET   = "Label"

DROP_PROJECTS = ["CodeIgniter"]   # 1 class - statistically meaningless

RANDOM_STATE = 42
CV_FOLDS     = 10
TEST_SIZE    = 0.20

# Bootstrap resamples for CI estimation [E2]
N_BOOTSTRAP = 1000

# Hyperparameter grid - thesis s3.4 / s4.2.2
# n_estimators in {100, 200, 500}
# max_depth    in {None, 10, 20}
# min_samples_leaf in {1, 2, 5}   <- thesis s4.2.2 reports best = min_samples_leaf=1
# class_weight in {balanced, None}
PARAM_GRID = {
    "rf__n_estimators":    [100, 200, 500],
    "rf__max_depth":       [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__class_weight":    ["balanced", None],
}

# Bonferroni-corrected alpha for 9 simultaneous Mann-Whitney tests [E5]
N_TESTS          = len(FEATURES)
ALPHA_RAW        = 0.05
ALPHA_BONFERRONI = ALPHA_RAW / N_TESTS   # = 0.0056

MAX_WATERFALL_PLOTS = 50

print("Configuration:")
print(f"  Features         : {FEATURES}")
print(f"  Target           : {TARGET}")
print(f"  CV folds         : {CV_FOLDS}")
print(f"  Test size        : {TEST_SIZE*100:.0f}%")
print(f"  Random state     : {RANDOM_STATE}")
print(f"  Bootstrap N      : {N_BOOTSTRAP}")
print(f"  alpha raw / Bonf. : {ALPHA_RAW} / {ALPHA_BONFERRONI:.4f}")
print(f"  SHAP available   : {SHAP_AVAILABLE}\n")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot=N_BOOTSTRAP,
                 alpha=0.05, random_state=RANDOM_STATE, **kwargs):
    """
    Bootstrap 95% CI for any sklearn metric. [E2]
    Returns (point_estimate, lower_ci, upper_ci).
    Uses the percentile method (Efron & Tibshirani, 1993).
    """
    rng = np.random.RandomState(random_state)
    point = metric_fn(y_true, y_pred if y_prob is None else y_prob, **kwargs)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            s = metric_fn(y_true[idx],
                          (y_pred if y_prob is None else y_prob)[idx],
                          **kwargs)
            scores.append(s)
        except Exception:
            pass
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def rank_biserial(u_stat, n1, n2):
    """
    Rank-biserial correlation from Mann-Whitney U. [E4]
    r = 1 - (2*U / (n1*n2))  ->  range [-1, 1]
    Interpretation: |r| >= 0.10 small, >= 0.30 medium, >= 0.50 large
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def cliffs_delta(group_a, group_b):
    """
    Cliff's Delta non-parametric effect size. [E9]
    delta = (# pairs where a > b  -  # pairs where a < b) / (n_a * n_b)
    Range [-1, 1]. Thresholds: |delta| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, >= 0.474 large (Romano et al., 2006).
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    dom = sum(1 if ai > bi else -1 if ai < bi else 0
              for ai in a for bi in b)
    return dom / (len(a) * len(b))


def ece_score(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error - mean |predicted_prob - empirical_freq|
    weighted by bin size. Lower = better calibrated. [E6]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask  = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos  = y_true[mask].mean()
        mean_prob = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(mean_prob - frac_pos)
    return ece



# ==============================================================================
# 2. DATA LOADING AND CLEANING  (s3.2)
# ==============================================================================

print("=" * 70)
print("STEP 1 - Data Loading, Cleaning & Label-Source Recording (s3.2)")
print("=" * 70)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Projects : {df['Project'].nunique()}")
print(f"Columns  : {list(df.columns)}\n")

# 2a. Drop single-class projects
before = len(df)
df = df[~df["Project"].isin(DROP_PROJECTS)]
print(f"Dropped projects {DROP_PROJECTS}: {before - len(df)} rows removed")

# 2b. Numeric coercion
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2c. Missing values
missing = df[FEATURES + [TARGET]].isnull().sum()
if missing.any():
    print(f"\nMissing values detected:\n{missing[missing > 0]}")
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Rows after dropping NaN: {len(df):,}")
else:
    print("No missing values found.")

# 2d. Label-source column (sensitivity analysis s3.2.2)
LABEL_SOURCE_COL = "LabelSource"
has_label_source = LABEL_SOURCE_COL in df.columns
if has_label_source:
    print(f"\nLabel source distribution:")
    print(df[LABEL_SOURCE_COL].value_counts().to_string())
else:
    print(f"\nNote: '{LABEL_SOURCE_COL}' column not found - "
          "sensitivity analysis (Step 14) will be skipped.")

# 2e. Dataset summary
df = df.reset_index(drop=True)   # clean integer index for iloc alignment
print(f"\nFinal dataset: {len(df):,} classes from {df['Project'].nunique()} projects")
lc = df[TARGET].value_counts()
print(f"Label distribution:")
print(f"  God Class     (1): {lc.get(1,0):>6,}  ({lc.get(1,0)/len(df)*100:.1f}%)")
print(f"  Non-God Class (0): {lc.get(0,0):>6,}  ({lc.get(0,0)/len(df)*100:.1f}%)")


# ==============================================================================
# 3. EDA + MANN-WHITNEY U TESTS + EFFECT SIZES + CORRECTIONS  (s3.3)  [E4,E5,E9]
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 2 - EDA + Mann-Whitney U + Effect Sizes + Bonferroni (s3.3)")
print("=" * 70)

god_mask    = df[TARGET] == 1
nongod_mask = df[TARGET] == 0

print("\nFeature means by label (God Class vs Non-God Class):")
print(df.groupby(TARGET)[FEATURES].mean().round(2).to_string())

print(f"\nMann-Whitney U tests (alpha_raw={ALPHA_RAW}, alpha_Bonferroni={ALPHA_BONFERRONI:.4f}):")
header = (f"\n  {'Metric':<8} {'U-stat':>12}  {'p-value':>12}  "
          f"{'Sig(raw)':>9}  {'Sig(Bonf)':>10}  "
          f"{'r_rb':>7}  {'Cliffs_d':>9}  {'Effect':>10}")
print(header)
print("  " + "-" * 85)

mwu_rows = []
for feat in FEATURES:
    god_vals    = df.loc[god_mask, feat].dropna().values
    nongod_vals = df.loc[nongod_mask, feat].dropna().values
    u_stat, p_val = stats.mannwhitneyu(god_vals, nongod_vals,
                                        alternative="two-sided")
    sig_raw  = p_val < ALPHA_RAW
    sig_bonf = p_val < ALPHA_BONFERRONI

    # Effect sizes [E4, E9]
    r_rb   = rank_biserial(u_stat, len(god_vals), len(nongod_vals))
    d_cliff = cliffs_delta(god_vals, nongod_vals)

    # Magnitude label (Cliff's delta thresholds)
    abs_d = abs(d_cliff)
    mag   = ("large"    if abs_d >= 0.474 else
             "medium"   if abs_d >= 0.330 else
             "small"    if abs_d >= 0.147 else "negligible")

    print(f"  {feat:<8} {u_stat:>12.1f}  {p_val:>12.4e}  "
          f"{'Yes' if sig_raw else 'No':>9}  "
          f"{'Yes' if sig_bonf else 'No':>10}  "
          f"{r_rb:>7.4f}  {d_cliff:>9.4f}  {mag:>10}")

    mwu_rows.append({
        "Feature": feat,
        "U_stat": u_stat, "p_value": p_val,
        "significant_raw": sig_raw,
        "significant_bonferroni": sig_bonf,
        "rank_biserial_r": round(r_rb, 4),
        "cliffs_delta": round(d_cliff, 4),
        "effect_magnitude": mag,
    })

mwu_df = pd.DataFrame(mwu_rows)
mwu_df.to_csv(OUTPUT_DIR / "mannwhitney_discriminability.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR}/mannwhitney_discriminability.csv")

discriminative_raw  = mwu_df[mwu_df["significant_raw"]]["Feature"].tolist()
discriminative_bonf = mwu_df[mwu_df["significant_bonferroni"]]["Feature"].tolist()
print(f"\n  Discriminative (p < {ALPHA_RAW})         : {discriminative_raw}")
print(f"  Discriminative (Bonferroni-corrected) : {discriminative_bonf}")

# Feature distributions with p-values on titles
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    god_v    = df[god_mask][feat]
    nongod_v = df[nongod_mask][feat]
    axes[i].hist(nongod_v.clip(upper=nongod_v.quantile(0.99)),
                 bins=40, alpha=0.6, label="Non-God", color="#4C9BE8", density=True)
    axes[i].hist(god_v.clip(upper=god_v.quantile(0.99)),
                 bins=40, alpha=0.6, label="God Class", color="#E8504C", density=True)
    row = mwu_df.set_index("Feature").loc[feat]
    axes[i].set_title(
        f"{feat}  (p={row['p_value']:.2e}, delta={row['cliffs_delta']:.3f} {row['effect_magnitude']})",
        fontsize=10)
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel("Value (clipped at P99)", fontsize=8)
fig.suptitle("Feature Distributions: God Class vs Non-God Class\n"
             "(p-values from Mann-Whitney U; delta = Cliff's delta)", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True, ax=ax,
            annot_kws={"size": 9})
ax.set_title("CK Metrics Feature Correlation Matrix", fontsize=13, pad=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/feature_distributions.png")
print(f"  Saved: {OUTPUT_DIR}/feature_correlation.png")


# ==============================================================================
# 4. TRAIN / TEST SPLIT  (s3.4)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 3 - Train / Test Split (Stratified Instance-Level 80/20, s3.4)")
print("=" * 70)

# Thesis s3.4: "The dataset is split into 80% training and 20% test using
# stratified sampling to preserve the God Class ratio in both subsets."
# Split is at the class (instance) level. Row indices are tracked for
# McNemar alignment and per-project analysis.
X = df[FEATURES].values
y = df[TARGET].values.astype(int)
indices = np.arange(len(df))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining set : {len(X_train):,} samples  "
      f"(God Class: {y_train.sum():,} = {y_train.mean()*100:.1f}%)")
print(f"Test set     : {len(X_test):,} samples   "
      f"(God Class: {y_test.sum():,} = {y_test.mean()*100:.1f}%)")


# ==============================================================================
# 5. PIPELINE CONSTRUCTION  (s3.3, s3.4)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 4 - Pipeline: StandardScaler -> SMOTE -> RF  (s3.3, s3.4)")
print("=" * 70)

# OOB score enabled as a zero-cost sanity check against CV results.
# oob_score requires bootstrap=True (sklearn default).
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ("rf", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
    )),
])

print("Pipeline stages:")
print("  1. StandardScaler   - Z-score normalization (train-fit only)")
print("  2. SMOTE            - minority oversampling in training folds only")
print("  3. RandomForest     - classifier with OOB scoring")


# ==============================================================================
# 6. HYPERPARAMETER TUNING - GridSearchCV  (s3.4)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 5 - Hyperparameter Tuning (GridSearchCV, macro-F1, s3.4)")
print("=" * 70)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=PARAM_GRID,
    cv=inner_cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

total_combos = 1
for v in PARAM_GRID.values():
    total_combos *= len(v)
print(f"Grid: {total_combos} combinations x {inner_cv.n_splits} inner folds\n")

grid_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best inner-CV macro-F1: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# Report OOB score from the best estimator (sanity check vs CV)
rf_best = best_pipeline.named_steps["rf"]
if hasattr(rf_best, "oob_score_"):
    print(f"\n  OOB Score: {rf_best.oob_score_:.4f}  "
          f"(sanity check - should be close to CV F1)")

gs_df = pd.DataFrame(grid_search.cv_results_)
gs_df.to_csv(OUTPUT_DIR / "gridsearch_results.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/gridsearch_results.csv")


# ==============================================================================
# 7. LEARNING CURVE ANALYSIS  [E7]
# ==============================================================================
# Shows whether performance improves with more data (training set size).
# If the curve has not plateaued, collecting more PHP classes would help.
# If it has plateaued, 19k classes is sufficient and adding data would not
# meaningfully improve results - an important finding for s5.5 future work.
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 6 - Learning Curve Analysis (s3.4, s5.5)")
print("=" * 70)

train_sizes_rel = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Computing learning curve (5-fold, macro-F1) ...")
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_pipeline,
    X_train, y_train,
    train_sizes=train_sizes_rel,
    cv=lc_cv,
    scoring="f1_macro",
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes_abs, train_mean, "o-", color="#E8504C", label="Training F1")
ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#E8504C")
ax.plot(train_sizes_abs, val_mean, "s-", color="#4C9BE8", label="Validation F1 (5-fold)")
ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color="#4C9BE8")
ax.set_xlabel("Training Set Size", fontsize=11)
ax.set_ylabel("Macro F1-Score", fontsize=11)
ax.set_title("Learning Curve - Random Forest God Class Detection\n"
             "(Plateau -> dataset size sufficient; Rising -> more data would help)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
plt.close()

# Quantify plateau: compare last two points
plateau_delta = abs(val_mean[-1] - val_mean[-2])
plateau_msg = "PLATEAU" if plateau_delta < 0.005 else "STILL RISING"
print(f"  Final validation F1: {val_mean[-1]:.4f} +/- {val_std[-1]:.4f}")
print(f"  Delta last two points: {plateau_delta:.4f} -> {plateau_msg}")
print(f"  Interpretation: {'Dataset size is sufficient for stable performance.'if plateau_delta < 0.005 else 'Performance is still improving - more data may help.'}")
print(f"  Saved: {OUTPUT_DIR}/learning_curve.png")

lc_df = pd.DataFrame({
    "train_size": train_sizes_abs,
    "train_f1_mean": train_mean, "train_f1_std": train_std,
    "val_f1_mean": val_mean, "val_f1_std": val_std,
})
lc_df.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)


# ==============================================================================
# 8. PROBABILITY CALIBRATION  [E6]
# ==============================================================================
# Random Forests are known to produce overconfident probabilities.
# Calibrating with isotonic regression maps raw probabilities to empirical
# frequencies. Well-calibrated probabilities are essential for:
#   (a) reliable threshold selection in Step 10
#   (b) valid SHAP base-rate interpretation (the base value = E[f(x)])
#   (c) Brier score comparison (proper scoring rule)
# Reliability diagram visualizes calibration quality.
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 7 - Probability Calibration (Isotonic Regression, s3.5)")
print("=" * 70)

# Calibrate on training set using 5-fold CV internally
calibrated_pipeline = CalibratedClassifierCV(
    best_pipeline,
    method="isotonic",   # isotonic > sigmoid (Platt) for large datasets
    cv=5,
)
calibrated_pipeline.fit(X_train, y_train)

y_prob_uncal  = best_pipeline.predict_proba(X_test)[:, 1]
y_prob_cal    = calibrated_pipeline.predict_proba(X_test)[:, 1]
y_pred_cal    = calibrated_pipeline.predict(X_test)

ece_uncal = ece_score(y_test, y_prob_uncal)
ece_cal   = ece_score(y_test, y_prob_cal)
brier_uncal = brier_score_loss(y_test, y_prob_uncal)
brier_cal   = brier_score_loss(y_test, y_prob_cal)

print(f"\n  Uncalibrated - ECE: {ece_uncal:.4f}  Brier: {brier_uncal:.4f}")
print(f"  Calibrated   - ECE: {ece_cal:.4f}  Brier: {brier_cal:.4f}")
print(f"  Calibration improvement: DeltaECE = {ece_uncal - ece_cal:+.4f}")

# Reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, probs, title in [
    (axes[0], y_prob_uncal, "Uncalibrated RF"),
    (axes[1], y_prob_cal,   "Calibrated RF (Isotonic)"),
]:
    fraction_of_positives, mean_predicted = calibration_curve(
        y_test, probs, n_bins=10, strategy="uniform"
    )
    ax.plot(mean_predicted, fraction_of_positives, "s-",
            color="#E8504C", label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram\n{title}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("Probability Calibration - God Class Detection", fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "calibration_reliability.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/calibration_reliability.png")


# ==============================================================================
# 10. 10-FOLD STRATIFIED CROSS-VALIDATION  (s3.5)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 8 - 10-Fold Stratified CV (s3.5): F1, Prec, Rec, AUC, MCC, PR-AUC")
print("=" * 70)

outer_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    best_pipeline,
    X_train, y_train,
    cv=outer_cv,
    scoring={
        "precision": "precision_macro",
        "recall":    "recall_macro",
        "f1":        "f1_macro",
        "roc_auc":   "roc_auc",
        "mcc":       "matthews_corrcoef",
        "pr_auc":    "average_precision",
    },
    return_train_score=False,
    n_jobs=-1,
)

print(f"\n10-Fold CV Results (mean +/- std, min, max across {CV_FOLDS} folds):")
print(f"  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print("  " + "-" * 55)
cv_metric_keys = ["f1", "precision", "recall", "roc_auc", "mcc", "pr_auc"]
for metric in cv_metric_keys:
    vals = cv_results[f"test_{metric}"]
    print(f"  {metric:<12}  {vals.mean():>8.4f}  {vals.std():>8.4f}  "
          f"{vals.min():>8.4f}  {vals.max():>8.4f}")

cv_df = pd.DataFrame({
    k.replace("test_", ""): v
    for k, v in cv_results.items() if k.startswith("test_")
})
cv_df.index = [f"Fold_{i+1}" for i in range(CV_FOLDS)]
cv_df.index.name = "fold"
cv_df.to_csv(OUTPUT_DIR / "cv_results.csv")

# CV boxplots (6 metrics)
fig, axes = plt.subplots(1, 6, figsize=(20, 5))
metric_labels = ["F1 (macro)", "Precision", "Recall", "ROC-AUC", "MCC", "PR-AUC"]
colors_plot   = ["#50C878", "#4C9BE8", "#E8504C", "#FF9500", "#9B59B6", "#1ABC9C"]
for ax, metric, label, color in zip(axes, cv_metric_keys, metric_labels, colors_plot):
    vals = cv_results[f"test_{metric}"]
    ax.boxplot(vals, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(vals.mean(), color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(1.15, vals.mean(), f"{vals.mean():.3f}", va="center", fontsize=8)
    ax.set_xticks([])
fig.suptitle(f"{CV_FOLDS}-Fold Cross-Validation Results", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cv_boxplots.png", dpi=150)
plt.close()

print(f"\n  Saved: {OUTPUT_DIR}/cv_results.csv, cv_boxplots.png")


# ==============================================================================
# 11. FINAL MODEL EVALUATION + BOOTSTRAP 95% CIs  (s3.5)  [E2]
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 9 - Final Evaluation on Held-Out Test Set + Bootstrap 95% CIs  (s3.5)")
print("=" * 70)

y_pred      = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

# Point estimates
f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
pr_auc    = average_precision_score(y_test, y_pred_prob)
mcc       = matthews_corrcoef(y_test, y_pred)
kappa     = cohen_kappa_score(y_test, y_pred)

# [E2] Bootstrap 95% confidence intervals
print("\nComputing bootstrap 95% CIs (1,000 resamples) ...")

ci_f1    = bootstrap_ci(y_test, y_pred, None,
                         lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
ci_prec  = bootstrap_ci(y_test, y_pred, None,
                         lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0))
ci_rec   = bootstrap_ci(y_test, y_pred, None,
                         lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0))
ci_auc   = bootstrap_ci(y_test, None, y_pred_prob,
                         lambda yt, yp: roc_auc_score(yt, yp))
ci_pr    = bootstrap_ci(y_test, None, y_pred_prob,
                         lambda yt, yp: average_precision_score(yt, yp))
ci_mcc   = bootstrap_ci(y_test, y_pred, None,
                         lambda yt, yp: matthews_corrcoef(yt, yp))

print(f"\nTest Set Metrics with 95% Bootstrap CIs:")
print(f"  {'Metric':<16} {'Point':>8}  {'95% CI':>20}  Notes")
print("  " + "-" * 70)
for name, pt, ci in [
    ("F1 (macro)",    f1,        ci_f1),
    ("Precision(mac)",precision, ci_prec),
    ("Recall (macro)",recall,    ci_rec),
    ("ROC-AUC",       roc_auc,   ci_auc),
    ("PR-AUC",        pr_auc,    ci_pr),
    ("MCC",           mcc,       ci_mcc),
]:
    print(f"  {name:<16} {pt:>8.4f}  [{ci[1]:.4f}, {ci[2]:.4f}]")

print(f"\n  Cohen's Kappa  : {kappa:.4f}")
print(f"  ECE (uncalib.) : {ece_uncal:.4f}")
print(f"  ECE (calib.)   : {ece_cal:.4f}")
print(f"  Brier (uncal.) : {brier_uncal:.4f}")
print(f"  Brier (calib.) : {brier_cal:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Non-God Class", "God Class"], digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}  FN: {fn:,}  TP: {tp:,}")

# Save all test metrics with CIs
test_metrics_df = pd.DataFrame([{
    "F1_macro": f1, "F1_macro_CI_lo": ci_f1[1], "F1_macro_CI_hi": ci_f1[2],
    "Precision_macro": precision, "Prec_CI_lo": ci_prec[1], "Prec_CI_hi": ci_prec[2],
    "Recall_macro": recall, "Rec_CI_lo": ci_rec[1], "Rec_CI_hi": ci_rec[2],
    "ROC_AUC": roc_auc, "AUC_CI_lo": ci_auc[1], "AUC_CI_hi": ci_auc[2],
    "PR_AUC": pr_auc, "PRAUC_CI_lo": ci_pr[1], "PRAUC_CI_hi": ci_pr[2],
    "MCC": mcc, "MCC_CI_lo": ci_mcc[1], "MCC_CI_hi": ci_mcc[2],
    "Cohens_Kappa": kappa,
    "ECE_uncalibrated": ece_uncal, "ECE_calibrated": ece_cal,
    "Brier_uncalibrated": brier_uncal, "Brier_calibrated": brier_cal,
    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
}])
test_metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

# Threshold analysis
thresh_rows = []
for t in np.arange(0.1, 0.91, 0.05):
    y_t = (y_pred_prob >= t).astype(int)
    if y_t.sum() == 0 or y_t.sum() == len(y_t):
        continue
    thresh_rows.append({
        "threshold": round(float(t), 2),
        "F1_macro":  round(f1_score(y_test, y_t, average="macro", zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_t, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_t, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_t), 4),
    })
thresh_df = pd.DataFrame(thresh_rows)
best_thresh = thresh_df.loc[thresh_df["F1_macro"].idxmax(), "threshold"]
print(f"\n  Default threshold (0.50) F1_macro: "
      f"{thresh_df[thresh_df['threshold']==0.50]['F1_macro'].values[0]:.4f}")
print(f"  Optimal threshold: {best_thresh:.2f} -> "
      f"F1_macro = {thresh_df[thresh_df['threshold']==best_thresh]['F1_macro'].values[0]:.4f}")
thresh_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

# Plots
for plot_fn, y_p, name, path in [
    (RocCurveDisplay.from_predictions,      y_pred_prob,
     f"Random Forest (AUC={roc_auc:.4f})",  "roc_curve.png"),
    (PrecisionRecallDisplay.from_predictions, y_pred_prob,
     f"Random Forest (PR-AUC={pr_auc:.4f})", "pr_curve.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_fn(y_test, y_p, name=name, ax=ax, color="#E8504C")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / path, dpi=150)
    plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Non-God Class", "God Class"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - Random Forest\nGod Class Detection (Test Set)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"\n  Saved: test_metrics.csv, roc_curve.png, pr_curve.png, "
      f"confusion_matrix.png, threshold_analysis.csv")


# ==============================================================================
# 12. McNemar's TEST + Cohen's g + FP/FN ANALYSIS  (s3.5)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 10 - McNemar's Test vs PHPMD + Cohen's g + FP/FN Analysis (s3.5)")
print("=" * 70)

if "PHPMD_Flag" not in df.columns:
    print("  WARNING: 'PHPMD_Flag' column not found - skipping McNemar's test.")
else:
    test_df = df.iloc[idx_test].copy()
    phpmd_pred = test_df["PHPMD_Flag"].values.astype(int)

    rf_correct    = (y_pred == y_test)
    phpmd_correct = (phpmd_pred == y_test)

    n = min(len(rf_correct), len(phpmd_correct))
    b = int(np.sum( rf_correct[:n] & ~phpmd_correct[:n]))
    c = int(np.sum(~rf_correct[:n] &  phpmd_correct[:n]))

    contingency = np.array([[0, b], [c, 0]])
    result = mcnemar(contingency, exact=True)

    p_wins   = b / (b + c) if (b + c) > 0 else np.nan
    cohens_g = abs(p_wins - 0.5) if not np.isnan(p_wins) else np.nan

    print(f"\nMcNemar's Test (exact binomial, alpha = 0.05):")
    print(f"  Discordant pairs (b+c) : {b + c}")
    print(f"  RF wins  (b)           : {b}")
    print(f"  PHPMD wins (c)         : {c}")
    print(f"  Discordant ratio b:c   : {b}:{c}  ({b/(b+c)*100:.1f}% RF wins)")
    print(f"  chi-sq statistic       : {result.statistic:.4f}")
    print(f"  p-value                : {result.pvalue:.4e}")
    print(f"  Cohen's g              : {cohens_g:.4f}  "
          f"({'large (>0.25)' if cohens_g > 0.25 else 'medium' if cohens_g > 0.15 else 'small'})")
    print(f"  Decision: {'REJECT H0 - RF significantly better than PHPMD' if result.pvalue < 0.05 else 'FAIL TO REJECT H0'}")

    phpmd_f1  = f1_score(y_test[:n], phpmd_pred[:n], average="macro", zero_division=0)
    phpmd_mcc = matthews_corrcoef(y_test[:n], phpmd_pred[:n])
    print(f"\nComparison on same test set:")
    print(f"  {'Metric':<12} {'RF':>8}  {'PHPMD':>8}  {'Delta':>8}")
    print(f"  {'F1 (macro)':<12} {f1:>8.4f}  {phpmd_f1:>8.4f}  {f1-phpmd_f1:>+8.4f}")
    print(f"  {'MCC':<12} {mcc:>8.4f}  {phpmd_mcc:>8.4f}  {mcc-phpmd_mcc:>+8.4f}")

    # FP/FN error pattern
    test_df["RF_pred"] = y_pred[:len(test_df)]
    test_df["RF_FP"] = ((test_df[TARGET]==0) & (test_df["RF_pred"]==1)).astype(int)
    test_df["RF_FN"] = ((test_df[TARGET]==1) & (test_df["RF_pred"]==0)).astype(int)
    fp_rows = test_df[test_df["RF_FP"]==1]
    fn_rows = test_df[test_df["RF_FN"]==1]
    print(f"\n  FP: {len(fp_rows)} (Non-God flagged as God)")
    print(f"  FN: {len(fn_rows)} (God Class missed)")
    if "Project" in test_df.columns and len(fp_rows) > 0:
        print("  Top FP projects:", fp_rows["Project"].value_counts().head(5).to_dict())

    pd.concat([fp_rows.assign(ErrorType="FP"), fn_rows.assign(ErrorType="FN")]
              ).to_csv(OUTPUT_DIR / "fp_fn_error_analysis.csv", index=False)

    pd.DataFrame([{
        "b_rf_wins": b, "c_phpmd_wins": c,
        "discordant_total": b+c, "statistic": result.statistic,
        "pvalue": result.pvalue, "significant_p05": result.pvalue < 0.05,
        "cohens_g": cohens_g,
        "phpmd_f1_macro": phpmd_f1, "rf_f1_macro": f1,
        "phpmd_mcc": phpmd_mcc, "rf_mcc": mcc,
    }]).to_csv(OUTPUT_DIR / "mcnemar_test.csv", index=False)
    print(f"  Saved: mcnemar_test.csv, fp_fn_error_analysis.csv")


# ==============================================================================
# 13. FEATURE IMPORTANCE: MDI + PERMUTATION  (s4.5.1)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 11 - Feature Importance: MDI + Permutation Cross-Check (s4.5.1)")
print("=" * 70)

scaler_step = best_pipeline.named_steps["scaler"]
rf_model    = best_pipeline.named_steps["rf"]

importances_mdi = rf_model.feature_importances_
std_mdi         = np.std([t.feature_importances_ for t in rf_model.estimators_], axis=0)

X_test_scaled = scaler_step.transform(X_test)
perm_result = permutation_importance(
    rf_model, X_test_scaled, y_test,
    n_repeats=30, random_state=RANDOM_STATE,
    scoring="f1_macro", n_jobs=-1,
)
perm_means = perm_result.importances_mean
perm_stds  = perm_result.importances_std

feat_imp_df = pd.DataFrame({
    "Feature":         FEATURES,
    "MDI_Importance":  importances_mdi,
    "MDI_Std":         std_mdi,
    "Perm_Importance": perm_means,
    "Perm_Std":        perm_stds,
}).sort_values("MDI_Importance", ascending=False).reset_index(drop=True)
feat_imp_df["Rank_MDI"]  = feat_imp_df["MDI_Importance"].rank(ascending=False).astype(int)
feat_imp_df["Rank_Perm"] = feat_imp_df["Perm_Importance"].rank(ascending=False).astype(int)

print(f"\n  {'Feature':<8}  {'MDI':>8}  {'Perm':>8}  {'Rank_MDI':>9}  {'Rank_Perm':>10}")
print("  " + "-" * 55)
for _, row in feat_imp_df.iterrows():
    print(f"  {row['Feature']:<8}  {row['MDI_Importance']:>8.4f}  "
          f"{row['Perm_Importance']:>8.4f}  {row['Rank_MDI']:>9}  {row['Rank_Perm']:>10}")

feat_imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Side-by-side MDI vs Permutation chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, std_col, title in [
    (axes[0], "MDI_Importance", "MDI_Std", "MDI (Mean Decrease in Impurity)"),
    (axes[1], "Perm_Importance", "Perm_Std", "Permutation Importance (Test Set, macro-F1)"),
]:
    df_s = feat_imp_df.sort_values(col, ascending=True)
    colors = ["#E8504C" if i >= len(df_s)-3 else "#4C9BE8" for i in range(len(df_s))]
    ax.barh(df_s["Feature"], df_s[col], xerr=df_s[std_col],
            color=colors, capsize=4, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("RF Feature Importance - God Class Detection (top 3 in red)", fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"  Saved: feature_importance.csv, feature_importance.png")


# ==============================================================================
# 14. SHAP TreeExplainer  (s3.4, s4.5)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 12 - SHAP TreeExplainer  (s3.4, s4.5)")
print("=" * 70)

if not SHAP_AVAILABLE:
    print("  SHAP not available - skipping.")
else:
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)

    print("  Computing SHAP TreeExplainer values ...")
    explainer = shap.TreeExplainer(rf_model)

    try:
        shap_values = explainer.shap_values(X_test_scaled_df)
        if isinstance(shap_values, list):
            shap_vals_2d = shap_values[1]
        else:
            shap_vals_2d = shap_values
        # If 3D (n_samples, n_features, n_classes), keep class 1
        if shap_vals_2d.ndim == 3:
            shap_vals_2d = shap_vals_2d[:, :, 1]
    except Exception as e:
        print(f"  ERROR computing SHAP values: {e}")
        shap_vals_2d = None

    if shap_vals_2d is not None:
        base_value = (float(explainer.expected_value)
                      if np.isscalar(explainer.expected_value)
                      else float(explainer.expected_value[1]))
        print(f"  Base rate (expected_value): {base_value:.4f}")
    else:
        base_value = 0.0

    # Global bar chart
    if shap_vals_2d is not None:
        mean_abs_shap = np.abs(shap_vals_2d).mean(axis=0)
        shap_global_df = pd.DataFrame({
            "Feature": FEATURES, "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#E8504C" if i < 3 else "#4C9BE8" for i in range(len(shap_global_df))]
        ax.barh(shap_global_df["Feature"], shap_global_df["Mean_Abs_SHAP"],
                color=colors, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Global Feature Importance (SHAP)\nGod Class Detection - PHP", fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(SHAP_DIR / "shap_global_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        shap_global_df.to_csv(SHAP_DIR / "shap_global_importance.csv", index=False)
        print(f"  Saved: {SHAP_DIR}/shap_global_bar.png")

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals_2d, X_test_scaled_df, show=False, plot_type="dot")
        plt.title("SHAP Beeswarm - God Class Predictions", fontsize=11)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {SHAP_DIR}/shap_beeswarm.png")

        # Dependence plots: WMC+CBO, LCOM+RFC
        for pf, ig in [("WMC", "CBO"), ("LCOM", "RFC")]:
            fig, ax = plt.subplots(figsize=(9, 6))
            shap.dependence_plot(pf, shap_vals_2d, X_test_scaled_df,
                                 interaction_index=ig, ax=ax, show=False)
            ax.set_title(f"SHAP Dependence: {pf} (colored by {ig})\n"
                         f"Non-linear threshold effects and interactions", fontsize=11)
            plt.tight_layout()
            fig.savefig(SHAP_DIR / f"shap_dependence_{pf.lower()}.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Interaction values RFC x CBO
        print("  Computing SHAP interaction values (RFC x CBO)...")
        try:
            shap_interaction = explainer.shap_interaction_values(X_test_scaled_df)
            if isinstance(shap_interaction, list):
                shap_interaction = shap_interaction[1]
            rfc_idx = FEATURES.index("RFC")
            cbo_idx = FEATURES.index("CBO")
            rfc_x_cbo = shap_interaction[:, rfc_idx, cbo_idx]
            print(f"  RFC x CBO interaction - mean={rfc_x_cbo.mean():.4f}, "
                  f"God Class mean={rfc_x_cbo[y_test==1].mean():.4f}")

            mean_interaction = np.abs(shap_interaction).mean(axis=0)
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(pd.DataFrame(mean_interaction, index=FEATURES, columns=FEATURES),
                        annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, annot_kws={"size": 8})
            ax.set_title("Mean |SHAP Interaction Values|\nRFC x CBO synergy highlighted", fontsize=12)
            plt.tight_layout()
            fig.savefig(SHAP_DIR / "shap_interaction_matrix.png", dpi=150, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(X_test_scaled_df["RFC"], X_test_scaled_df["CBO"],
                            c=rfc_x_cbo, cmap="RdBu_r", alpha=0.5, s=10)
            plt.colorbar(sc, ax=ax, label="SHAP Interaction (RFC x CBO)")
            ax.set_xlabel("RFC (scaled)"); ax.set_ylabel("CBO (scaled)")
            ax.set_title("RFC x CBO SHAP Interaction\n"
                         "Positive = synergistic push toward God Class", fontsize=11)
            plt.tight_layout()
            fig.savefig(SHAP_DIR / "shap_interaction_rfc_cbo.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  WARNING: SHAP interaction values failed ({e}).")

        # Per-class waterfall plots
        flagged_idx = np.where(y_pred == 1)[0]
        n_plots = min(len(flagged_idx), MAX_WATERFALL_PLOTS)
        print(f"\n  Generating {n_plots} waterfall plots.")

        for plot_i, test_i in enumerate(flagged_idx[:n_plots]):
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals_2d[test_i],
                        base_values=base_value,
                        data=X_test_scaled_df.iloc[test_i].values,
                        feature_names=FEATURES,
                    ),
                    show=False,
                    max_display=9
                )
                true_label = (
                    "God Class"
                    if y_test[test_i] == 1
                    else "Non-God Class"
                )
                plt.title(
                    f"SHAP Waterfall - Test instance #{test_i}\n"
                    f"True: {true_label} | Pred prob: {y_pred_prob[test_i]:.3f}",
                    fontsize=10
                )
                plt.tight_layout()
                plt.savefig(
                    SHAP_DIR / f"waterfall_{plot_i:04d}_inst{test_i}.png",
                    dpi=120,
                    bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                print(f"  WARNING: waterfall plot failed ({e})")

        print(f"  Saved {n_plots} waterfall plots to {SHAP_DIR}/")


# ==============================================================================
# 15. SENSITIVITY ANALYSIS - Label Source  (s3.2.2)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 13 - Sensitivity Analysis: Label Source (s3.2.2)")
print("=" * 70)

if not has_label_source:
    print("  Skipped - 'LabelSource' column not present.")
else:
    source_results = []
    for source in ["PHPMD-confirmed", "expert-overridden", "dual-source"]:
        mask_source = df[LABEL_SOURCE_COL] == source
        n_source    = mask_source.sum()
        if n_source < 20:
            continue
        sub_df = df[mask_source | (df[TARGET] == 0)]
        X_sub  = sub_df[FEATURES].values
        y_sub  = sub_df[TARGET].values.astype(int)
        if y_sub.mean() < 0.01 or y_sub.mean() > 0.99:
            continue
        X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
            X_sub, y_sub, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y_sub)
        sub_pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("rf",     RandomForestClassifier(**{
                k.replace("rf__", ""): v
                for k, v in grid_search.best_params_.items()
            }, random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        sub_pipe.fit(X_s_tr, y_s_tr)
        y_s_pred = sub_pipe.predict(X_s_te)
        f1_s  = f1_score(y_s_te, y_s_pred, average="macro", zero_division=0)
        mcc_s = matthews_corrcoef(y_s_te, y_s_pred)
        source_results.append({
            "LabelSource": source, "N": n_source,
            "F1_macro": round(f1_s, 4), "MCC": round(mcc_s, 4)
        })
        print(f"  {source:<25} N={n_source:>5}  F1_macro={f1_s:.4f}  MCC={mcc_s:.4f}")

    if source_results:
        pd.DataFrame(source_results).to_csv(
            OUTPUT_DIR / "sensitivity_label_source.csv", index=False)
        print(f"  Saved: sensitivity_label_source.csv")


# ==============================================================================
# 16. PER-PROJECT GENERALIZATION ANALYSIS  (s4.7)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 14 - Per-Project Generalization Analysis  (s4.7)")
print("=" * 70)

test_df_pp = df.iloc[idx_test].copy()
test_df_pp["y_pred"]      = y_pred[:len(test_df_pp)]
test_df_pp["y_pred_prob"] = y_pred_prob[:len(test_df_pp)]

per_proj = []
for proj, group in test_df_pp.groupby("Project"):
    if group[TARGET].nunique() < 2:
        continue
    p = precision_score(group[TARGET], group["y_pred"], zero_division=0)
    r = recall_score(group[TARGET], group["y_pred"], zero_division=0)
    f = f1_score(group[TARGET], group["y_pred"], zero_division=0)
    m = matthews_corrcoef(group[TARGET], group["y_pred"]) \
        if group[TARGET].nunique() > 1 else float("nan")
    per_proj.append({
        "Project": proj, "N_classes": len(group),
        "N_godclass": int(group[TARGET].sum()),
        "Prevalence": round(group[TARGET].mean(), 4),
        "Precision": round(p, 4), "Recall": round(r, 4),
        "F1": round(f, 4), "MCC": round(m, 4),
    })

per_proj_df = pd.DataFrame(per_proj).sort_values("F1", ascending=False)
print(f"\n{'Project':<25} {'N':>6} {'GC':>5} {'Prev':>7} "
      f"{'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8}")
print("  " + "-" * 80)
for _, row in per_proj_df.iterrows():
    print(f"  {row['Project']:<23} {row['N_classes']:>6} {row['N_godclass']:>5} "
          f"{row['Prevalence']:>7.3f} {row['Precision']:>8.4f} "
          f"{row['Recall']:>8.4f} {row['F1']:>8.4f} {row['MCC']:>8.4f}")

per_proj_df.to_csv(OUTPUT_DIR / "per_project_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 6))
colors_pp = ["#E8504C" if fv < 0.75 else "#50C878" for fv in per_proj_df["F1"]]
ax.barh(per_proj_df["Project"], per_proj_df["F1"],
        color=colors_pp, edgecolor="white")
ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, label="F1 = 0.75 threshold")
ax.axvline(f1, color="#FF9500", linestyle="--", alpha=0.7,
           label=f"Overall F1 = {f1:.4f}")
ax.set_xlabel("F1 Score"); ax.set_title("Per-Project F1 - God Class Detection")
ax.legend(fontsize=9); ax.set_xlim(0, 1.05); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "per_project_f1.png", dpi=150)
plt.close()
print(f"\n  Saved: per_project_metrics.csv, per_project_f1.png")


# ==============================================================================
# 17. MODEL PERSISTENCE + REPRODUCIBILITY MANIFEST  [E10]
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 15 - Model Persistence + Reproducibility Manifest")
print("=" * 70)

# Save trained pipeline
model_path = OUTPUT_DIR / "godclass_rf_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
print(f"  Saved: {model_path}")

# Also save calibrated pipeline
cal_model_path = OUTPUT_DIR / "godclass_rf_calibrated.joblib"
joblib.dump(calibrated_pipeline, cal_model_path)
print(f"  Saved: {cal_model_path}")

# Reproducibility manifest - JSON record for full experiment traceability
manifest = {
    "timestamp": datetime.now().isoformat(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "random_state": RANDOM_STATE,
    "cv_folds": CV_FOLDS,
    "test_size": TEST_SIZE,
    "n_bootstrap": N_BOOTSTRAP,
    "alpha_raw": ALPHA_RAW,
    "alpha_bonferroni": ALPHA_BONFERRONI,
    "dataset": {
        "total_classes": len(df),
        "god_classes": int(df[TARGET].sum()),
        "projects": df["Project"].nunique(),
        "features": FEATURES,
    },
    "best_hyperparameters": grid_search.best_params_,
    "test_metrics": {
        "f1_macro":      round(f1, 4),
        "f1_95ci":       [round(ci_f1[1], 4), round(ci_f1[2], 4)],
        "roc_auc":       round(roc_auc, 4),
        "auc_95ci":      [round(ci_auc[1], 4), round(ci_auc[2], 4)],
        "mcc":           round(mcc, 4),
        "mcc_95ci":      [round(ci_mcc[1], 4), round(ci_mcc[2], 4)],
        "pr_auc":        round(pr_auc, 4),
        "cohens_kappa":  round(kappa, 4),
        "ece_uncal":     round(ece_uncal, 4),
        "ece_cal":       round(ece_cal, 4),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn),
                             "FP": int(fp), "FN": int(fn)},
    },
    "discriminability": {
        row["Feature"]: {
            "p_value": row["p_value"],
            "significant_bonferroni": bool(row["significant_bonferroni"]),
            "cliffs_delta": row["cliffs_delta"],
            "effect_magnitude": row["effect_magnitude"],
        }
        for _, row in mwu_df.iterrows()
    },
    "library_versions": {
        "numpy":        np.__version__,
        "pandas":       pd.__version__,
        "sklearn":      __import__("sklearn").__version__,
        "imbalanced_learn": __import__("imblearn").__version__,
        "shap":         __import__("shap").__version__ if SHAP_AVAILABLE else "not installed",
        "statsmodels":  __import__("statsmodels").__version__,
    },
    "runtime_seconds": round(time.time() - RUN_START, 1),
}

manifest_path = OUTPUT_DIR / "reproducibility_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"  Saved: {manifest_path}")


# ==============================================================================
# 16. FINAL SUMMARY
# ==============================================================================

logging.info("Experiment completed successfully")
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

runtime = time.time() - RUN_START
print(f"""
Dataset
  Total classes    : {len(df):,}
  God Classes      : {int(df[TARGET].sum()):,} ({df[TARGET].mean()*100:.1f}%)
  Projects         : {df['Project'].nunique()}
  Features used    : {len(FEATURES)} ({', '.join(FEATURES)})
  Split type       : Stratified instance-level (80/20, s3.4)

Discriminability (Mann-Whitney U, s3.3)
  Significant (raw alpha={ALPHA_RAW})         : {discriminative_raw}
  Significant (Bonferroni alpha={ALPHA_BONFERRONI:.4f}): {discriminative_bonf}

Model Performance - Held-Out Test Set (s3.5)
  F1 (macro)       : {f1:.4f}  95% CI [{ci_f1[1]:.4f}, {ci_f1[2]:.4f}]
  ROC-AUC          : {roc_auc:.4f}  95% CI [{ci_auc[1]:.4f}, {ci_auc[2]:.4f}]
  PR-AUC           : {pr_auc:.4f}  95% CI [{ci_pr[1]:.4f}, {ci_pr[2]:.4f}]
  MCC              : {mcc:.4f}  95% CI [{ci_mcc[1]:.4f}, {ci_mcc[2]:.4f}]
  Cohen's Kappa    : {kappa:.4f}
  ECE (calibrated) : {ece_cal:.4f}

10-Fold CV (s3.5)
  F1   mean +/- std  : {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}
  AUC  mean +/- std  : {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}
  MCC  mean +/- std  : {cv_results['test_mcc'].mean():.4f} +/- {cv_results['test_mcc'].std():.4f}

Learning Curve (s5.5)
  Final val F1     : {val_mean[-1]:.4f} +/- {val_std[-1]:.4f}
  Plateau status   : {plateau_msg}

Output Directory : {OUTPUT_DIR}
  KEY FILES:
  reproducibility_manifest.json      - full experiment record
  test_metrics.csv                   - all metrics with 95% CIs
  mannwhitney_discriminability.csv   - U, p, r_rb, Cliffs delta (s3.3)
  learning_curve.csv / .png          - data sufficiency (s5.5)
  calibration_reliability.png        - probability calibration
  feature_importance.csv             - MDI + permutation (s4.5.1)
  gridsearch_results.csv             - all hyperparameter combos
  cv_results.csv                     - fold-level CV metrics
  mcnemar_test.csv                   - PHPMD comparison (s3.5)
  fp_fn_error_analysis.csv           - error patterns
  per_project_metrics.csv            - per-project F1 + MCC (s4.7)
  shap_global_bar.png / beeswarm.png / dependence_*.png  (s4.5)
  shap_interaction_matrix.png / rfc_cbo.png
  shap/waterfall_*.png               - per-class refactoring guidance
  godclass_rf_pipeline.joblib        - trained model
  godclass_rf_calibrated.joblib      - calibrated model

Runtime: {runtime:.0f} seconds
""")
print("DONE - All outputs written to", OUTPUT_DIR)


# ==============================================================================
# GLOBAL ERROR HANDLER
# ==============================================================================
# Note: To use this handler, wrap the full script in a try/except.
# Run with: python train_godclass.py > results.txt 2>&1
# Any fatal crash will be saved to fatal_error_log.txt in the working directory.
# ==============================================================================
# except Exception as e:
#     import traceback
#     print("\nFATAL ERROR:")
#     print(str(e))
#     with open("fatal_error_log.txt", "w", encoding="utf-8") as _f:
#         traceback.print_exc(file=_f)
#     raise