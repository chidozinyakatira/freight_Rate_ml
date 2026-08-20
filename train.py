"""
Train and validate the freight rate model.

Validation strategy
--------------------
The task is fundamentally a forecasting problem: we have labeled loads from
Jan-Oct 2025 and must predict unseen loads in Nov-Dec 2025. A random
train/test split would let the model see loads from *after* the dates in
its "test" fold, which overstates how well it will generalize forward in
time (and can leak lane/date-level patterns across the split).

Instead we use a chronological holdout:
  - Train fold:      2025-01-01 to 2025-08-31  (~8 months)
  - Holdout fold:     2025-09-01 to 2025-10-31  (~2 months, most recent)

This holdout is used purely to pick the model/hyperparameters and to report
honest error metrics. Once validated, the FINAL model is refit on the full
labeled dataset (Jan-Oct) before scoring validation.csv, since there is no
reason to withhold Sep/Oct data from the model we actually ship.

Model
-----
XGBoost gradient-boosted trees. Rationale (see report):
  - Tabular data with a mix of continuous (distance, weight, market_index,
    quote_signal) and categorical (equipment, pickup/delivery city)
    features, and a highly non-linear/interacting target (rate depends on
    distance together with equipment together with lane, not additively).
  - Handles the mild missingness gracefully alongside the explicit
    imputation step, and needs no feature scaling.
  - Fast to train/tune on 48k rows and gives feature importances that are
    easy to sanity-check against domain intuition (distance should
    dominate, which it does).
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from features import Encoder, FEATURE_COLUMNS, Imputer, add_calendar_features, add_distance_features, basic_clean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train_test.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "posted_rate"
SPLIT_DATE = "2025-09-01"

# ~1.3% of rows have a rate-per-mile far outside the normal band (0.9-5.0
# $/mi vs a median of ~2.15) with no distinguishing feature (spread evenly
# across equipment, lane, and month) - see report for detail. Treated as
# unexplained anomalies/noise and EXCLUDED FROM TRAINING ONLY. They are kept
# in the holdout fold so reported error metrics stay honest about the loads
# the model cannot explain.
RPM_LOW, RPM_HIGH = 0.9, 5.0

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)


def prep(df, imputer, encoder):
    df = basic_clean(df)
    df = imputer.transform(df)
    df = add_calendar_features(df)
    df = add_distance_features(df)
    df = encoder.transform(df)
    return df


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE=${mae:,.2f}  RMSE=${rmse:,.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def main():
    raw = pd.read_csv(DATA)
    raw = basic_clean(raw)

    train_mask = raw["date"] < SPLIT_DATE
    train_raw, holdout_raw = raw[train_mask].copy(), raw[~train_mask].copy()
    print(f"Train fold: {len(train_raw)} rows ({train_raw.date.min().date()} to {train_raw.date.max().date()})")
    print(f"Holdout fold: {len(holdout_raw)} rows ({holdout_raw.date.min().date()} to {holdout_raw.date.max().date()})")

    rpm = train_raw[TARGET] / train_raw["distance"]
    anomaly_mask = ~rpm.between(RPM_LOW, RPM_HIGH)
    print(f"Dropping {anomaly_mask.sum()} rate-per-mile anomalies from the training fold (kept in holdout)")
    train_raw = train_raw[~anomaly_mask].copy()

    # --- fit preprocessing on TRAIN fold only, evaluate on holdout ---
    imputer = Imputer().fit(train_raw)
    encoder = Encoder().fit(train_raw)  # holdout cities/equipment are a subset of train's

    train_feat = prep(train_raw, imputer, encoder)
    holdout_feat = prep(holdout_raw, imputer, encoder)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train_feat[FEATURE_COLUMNS], train_feat[TARGET],
        eval_set=[(holdout_feat[FEATURE_COLUMNS], holdout_feat[TARGET])],
        verbose=False,
    )

    holdout_pred = model.predict(holdout_feat[FEATURE_COLUMNS])
    holdout_pred = np.clip(holdout_pred, 1, None)
    metrics = evaluate(holdout_feat[TARGET], holdout_pred, "Holdout (Sep-Oct 2025)")

    train_pred = model.predict(train_feat[FEATURE_COLUMNS])
    evaluate(train_feat[TARGET], train_pred, "Train fold (sanity check)")

    # feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    # --- diagnostic plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(holdout_feat[TARGET], holdout_pred, s=6, alpha=0.3)
    lims = [0, max(holdout_feat[TARGET].max(), holdout_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Actual posted_rate")
    axes[0].set_ylabel("Predicted posted_rate")
    axes[0].set_title("Holdout: Predicted vs Actual")

    resid = holdout_pred - holdout_feat[TARGET].values
    axes[1].hist(resid, bins=60)
    axes[1].set_title("Holdout residuals (pred - actual)")
    axes[1].set_xlabel("Residual ($)")
    fig.tight_layout()
    fig.savefig(OUT / "holdout_diagnostics.png", dpi=150)
    print(f"\nSaved diagnostic plot to {OUT / 'holdout_diagnostics.png'}")

    with open(OUT / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- refit final model on ALL labeled data (Jan-Oct) for production use ---
    full_rpm = raw[TARGET] / raw["distance"]
    full_clean = raw[full_rpm.between(RPM_LOW, RPM_HIGH)].copy()
    print(f"\nFinal refit: dropping {len(raw) - len(full_clean)} anomalies, training on {len(full_clean)} rows")

    full_imputer = Imputer().fit(full_clean)
    full_encoder = Encoder().fit(full_clean)
    full_feat = prep(full_clean, full_imputer, full_encoder)

    final_params = dict(XGB_PARAMS)
    final_params["n_estimators"] = model.best_iteration + 1 if hasattr(model, "best_iteration") and model.best_iteration else XGB_PARAMS["n_estimators"]
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(full_feat[FEATURE_COLUMNS], full_feat[TARGET], verbose=False)

    joblib.dump(
        {"model": final_model, "imputer": full_imputer, "encoder": full_encoder, "features": FEATURE_COLUMNS},
        OUT / "model.joblib",
    )
    print(f"Saved final production model (trained on all {len(raw)} labeled rows) to {OUT / 'model.joblib'}")


if __name__ == "__main__":
    main()
