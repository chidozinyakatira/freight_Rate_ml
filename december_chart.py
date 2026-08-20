"""
Fill data/december_chart_inputs.csv's predicted_rate column.

This file is a controlled scenario, NOT a slice of real loads: every row
holds pickup, delivery, distance, equipment, and weight IDENTICAL (Lexington
-> Fort Wayne, 360 mi, Dry Van, 32,000 lb) across all 31 days of December
2025 - only `date` changes row to row. Its purpose is to isolate the model's
learned date/seasonality effect from every other variable, since in every
other artifact in this project distance/equipment/lane/weight also vary and
would confound a "what does the model think December looks like" question.

market_index / quote_signal are not columns in this file (the scorer's
`score.py` only recognizes the 7 columns already present), so they can't
vary by date here either - consistent with the file's "only date changes"
design. Both are filled ONCE with a single fixed value: the median
market_index / quote_signal historically observed on the Lexington -> Fort
Wayne, Dry Van lane in data/train_test.csv (21 matching rows). This keeps
every row's inputs identical except date, exactly as the file's fixed
columns already do, and gives the model a plausible "current market
conditions" reading for this specific lane rather than an arbitrary global
default.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import add_calendar_features, add_distance_features, basic_clean

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "model.joblib"
TRAIN_PATH = ROOT / "data" / "train_test.csv"
INPUT_PATH = ROOT / "data" / "december_chart_inputs.csv"
OUT_PATH = ROOT / "outputs" / "december_chart_inputs.csv"


def main():
    bundle = joblib.load(MODEL_PATH)
    model, imputer, encoder, feature_cols = bundle["model"], bundle["imputer"], bundle["encoder"], bundle["features"]

    dec = pd.read_csv(INPUT_PATH)
    assert list(dec.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"], (
        "december_chart_inputs.csv columns/order changed - check against the scorer's expected schema"
    )

    pickup, delivery, equipment = dec.loc[0, ["pickup", "delivery", "equipment"]]
    train = pd.read_csv(TRAIN_PATH)
    lane_equip = train[(train.pickup == pickup) & (train.delivery == delivery) & (train.equipment == equipment)]
    print(f"Lane {pickup} -> {delivery}, {equipment}: {len(lane_equip)} historical loads used for market signal medians")

    market_index_fill = lane_equip["market_index"].median()
    quote_signal_fill = lane_equip["quote_signal"].median()
    print(f"market_index fill = {market_index_fill:.5f}, quote_signal fill = {quote_signal_fill:.5f}")

    dec_clean = basic_clean(dec)
    dec_clean["market_index"] = market_index_fill
    dec_clean["quote_signal"] = quote_signal_fill
    dec_clean = imputer.transform(dec_clean)  # no-op here since both are already filled; kept for pipeline consistency
    dec_clean = add_calendar_features(dec_clean)
    dec_clean = add_distance_features(dec_clean)
    dec_clean = encoder.transform(dec_clean)

    preds = model.predict(dec_clean[feature_cols])
    preds = np.clip(preds, 1, None)

    dec_out = dec.copy()
    dec_out["predicted_rate"] = np.round(preds, 2)
    dec_out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(dec_out)} rows to {OUT_PATH}")
    print(dec_out[["date", "predicted_rate"]])


if __name__ == "__main__":
    main()
