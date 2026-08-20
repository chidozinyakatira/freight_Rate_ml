"""
Score every load in data/validation.csv with the trained model and write
validation_predictions.csv with columns: load_id,predicted_rate

Run `train.py` first to produce outputs/model.joblib.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import add_calendar_features, add_distance_features, basic_clean

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "model.joblib"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
OUT_PATH = ROOT / "outputs" / "validation_predictions.csv"


def main():
    bundle = joblib.load(MODEL_PATH)
    model, imputer, encoder, feature_cols = bundle["model"], bundle["imputer"], bundle["encoder"], bundle["features"]

    val = pd.read_csv(VALIDATION_PATH)
    val_clean = basic_clean(val)
    val_clean = imputer.transform(val_clean)
    val_clean = add_calendar_features(val_clean)
    val_clean = add_distance_features(val_clean)
    val_clean = encoder.transform(val_clean)

    preds = model.predict(val_clean[feature_cols])
    preds = np.clip(preds, 1, None)  # rates can't be <= 0

    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(preds, 2)})
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} predictions to {OUT_PATH}")
    print(out.describe())


if __name__ == "__main__":
    main()
