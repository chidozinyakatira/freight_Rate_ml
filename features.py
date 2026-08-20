"""
Feature engineering and cleaning shared by training and prediction.

Design notes
------------
- `weight` contains a small number of negative values (292 in train_test.csv,
  ~0.6% of rows). These have the exact same magnitude distribution as valid
  weights, just with a flipped sign -> treated as a sign-entry error and
  corrected with abs(), not dropped.
- `weight` and `market_index` have a small fraction of missing values
  (~0.6-0.8%). Both are imputed using the median computed WITHIN the
  training fold only (equipment-level median for weight, lane-level median
  for market_index), to avoid leaking validation-fold information into the
  imputation statistic.
- `distance` is fully populated and is by far the strongest single driver of
  `posted_rate` (corr ~0.91), so it is used as-is plus a log transform to
  help the model capture the mild sub-linear relationship at very long
  hauls.
- `market_index` and `quote_signal` are lane/row-specific external signals
  (not simple date-level averages - each row on the same date has a
  different value), so they are kept as direct numeric features. They are
  the main channel through which the model can pick up real Nov/Dec market
  conditions, since actual (not synthetic) values for those signals are
  present in validation.csv.
- Calendar features (month, day-of-week, day-of-year) capture the mild
  seasonal drift in rate-per-mile observed in train_test.csv (~2.10 in
  Jan rising to ~2.33 in Jun, easing back down). Because Nov/Dec are
  outside the training month range, month is encoded cyclically (sin/cos)
  so the model treats Dec as numerically adjacent to Jan rather than a
  wholly novel category.
- `pickup`/`delivery` city names are label-encoded (not one-hot, to keep
  the feature count sane) and handed to XGBoost as native categorical
  columns, letting the tree model learn lane-specific effects without an
  explicit target-encoding step (which would risk leakage).
"""

import numpy as np
import pandas as pd

RAW_NUMERIC = ["distance", "weight", "market_index", "quote_signal"]
CATEGORICAL = ["equipment", "pickup", "delivery"]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues. Does not impute (see `fit_impute`)."""
    df = df.copy()
    # Sign-entry error: negative weights -> take absolute value.
    df["weight"] = df["weight"].abs()
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log1p(df["distance"])
    return df


class Imputer:
    """Learns imputation statistics on a training fold, applies elsewhere."""

    def fit(self, df: pd.DataFrame):
        self.weight_by_equipment_ = df.groupby("equipment")["weight"].median()
        self.weight_global_ = df["weight"].median()
        self.market_by_lane_ = df.groupby(["pickup", "delivery"])["market_index"].median()
        self.market_global_ = df["market_index"].median()
        # quote_signal was never missing in train_test.csv, but the December
        # chart input file (fixed-lane, date-only scenario) doesn't carry
        # market_index/quote_signal at all, so the same lane+equipment
        # median lookup is reused there. Fit here for consistency even
        # though transform() below doesn't need it for train/validation.csv.
        self.quote_by_lane_equip_ = df.groupby(["pickup", "delivery", "equipment"])["quote_signal"].median()
        self.quote_by_lane_ = df.groupby(["pickup", "delivery"])["quote_signal"].median()
        self.quote_global_ = df["quote_signal"].median()
        self.market_by_lane_equip_ = df.groupby(["pickup", "delivery", "equipment"])["market_index"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        need_w = df["weight"].isna()
        if need_w.any():
            fill = df.loc[need_w, "equipment"].map(self.weight_by_equipment_)
            fill = fill.fillna(self.weight_global_)
            df.loc[need_w, "weight"] = fill

        if "market_index" not in df.columns:
            df["market_index"] = np.nan
        need_m = df["market_index"].isna()
        if need_m.any():
            lane_key = list(zip(df.loc[need_m, "pickup"], df.loc[need_m, "delivery"]))
            fill = pd.Series(
                [self.market_by_lane_.get(k, np.nan) for k in lane_key],
                index=df.loc[need_m].index,
            )
            fill = fill.fillna(self.market_global_)
            df.loc[need_m, "market_index"] = fill

        if "quote_signal" not in df.columns:
            df["quote_signal"] = np.nan
        need_q = df["quote_signal"].isna()
        if need_q.any():
            lane_equip_key = list(zip(
                df.loc[need_q, "pickup"], df.loc[need_q, "delivery"], df.loc[need_q, "equipment"]
            ))
            fill = pd.Series(
                [self.quote_by_lane_equip_.get(k, np.nan) for k in lane_equip_key],
                index=df.loc[need_q].index,
            )
            lane_key = list(zip(df.loc[need_q, "pickup"], df.loc[need_q, "delivery"]))
            lane_fallback = pd.Series(
                [self.quote_by_lane_.get(k, np.nan) for k in lane_key],
                index=df.loc[need_q].index,
            )
            fill = fill.fillna(lane_fallback).fillna(self.quote_global_)
            df.loc[need_q, "quote_signal"] = fill
        return df


class Encoder:
    """Label-encodes categorical columns; unseen categories map to a
    reserved 'unknown' code rather than raising."""

    def fit(self, df: pd.DataFrame):
        self.maps_ = {}
        for col in CATEGORICAL:
            cats = sorted(df[col].astype(str).unique())
            self.maps_[col] = {c: i for i, c in enumerate(cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL:
            m = self.maps_[col]
            unknown_code = len(m)
            df[col + "_code"] = df[col].astype(str).map(m).fillna(unknown_code).astype(int)
        return df


FEATURE_COLUMNS = [
    "distance",
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "month_sin",
    "month_cos",
    "day_of_week",
    "day_of_year",
    "equipment_code",
    "pickup_code",
    "delivery_code",
]


def build_features(df: pd.DataFrame, imputer: Imputer, encoder: Encoder) -> pd.DataFrame:
    df = basic_clean(df)
    df = imputer.transform(df)
    df = add_calendar_features(df)
    df = add_distance_features(df)
    df = encoder.transform(df)
    return df
