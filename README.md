# Freight Rate Prediction

Predicts `posted_rate` for freight loads using an XGBoost gradient-boosted
tree model trained on `data/train_test.csv`, then scores every load in
`data/validation.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
data/                   train_test.csv, validation.csv
notebook/
  freight_rate_prediction.ipynb   full walkthrough: EDA -> cleaning -> feature
                                    engineering -> split -> train -> validate ->
                                    refit -> score validation.csv -> Dec chart
                                    (pre-run, all cells have output)
src/
  features.py            shared cleaning / feature engineering / imputation
  train.py                time-based split, trains + validates + saves model
  predict.py               scores validation.csv -> validation_predictions.csv
outputs/
  model.joblib             trained model + fitted preprocessors (created by train.py)
  holdout_diagnostics.png  predicted-vs-actual + residuals on the holdout fold
  holdout_metrics.json     MAE / RMSE / MAPE / R2 on the holdout fold
  validation_predictions.csv   final submission file
  december_predicted_rate_chart.png   mean predicted rate by day, Dec 2025
  report.docx              written report (approach, findings, chart)
```

## Run

Two equivalent ways to reproduce everything — pick one.

**Notebook (recommended for review / the code walkthrough):**

```bash
jupyter notebook notebook/freight_rate_prediction.ipynb
```
Copy `data/train_test.csv` and `data/validation.csv` alongside the notebook
first (or adjust the `TRAIN_PATH` / `VALIDATION_PATH` constants in the first
code cell), then Run All. It reproduces every step end-to-end, including
`validation_predictions.csv` and the December chart, with all outputs
already baked into the committed copy so it can be reviewed without
re-running.

**Scripts (recommended for CI / one-shot regeneration):**

```bash
cd src
python train.py      # trains on Jan-Aug, validates on Sep-Oct, then refits
                      # on all labeled data and saves outputs/model.joblib
python predict.py    # scores data/validation.csv, writes
                      # outputs/validation_predictions.csv
```

Both paths use the identical cleaning / feature engineering logic and
produce bit-identical predictions.

See the accompanying report (`report.docx`) for the validation methodology,
data-quality findings, and model rationale.

## Note on the assessment's `score.py` / `december_chart_inputs.csv`

Both are included in this repo (`src/score.py` is the scorer as provided;
`data/december_chart_inputs.csv` is the original unfilled template). Run:

```bash
cd src
python train.py             # produces outputs/model.joblib
python predict.py           # produces outputs/validation_predictions.csv
python december_chart.py    # fills december_chart_inputs.csv -> outputs/december_chart_inputs.csv
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs/scorer_results
```

`december_chart_inputs.csv` holds pickup, delivery, distance, equipment, and
weight identical across all 31 December rows — only `date` changes. It has
no `market_index`/`quote_signal` columns (the scorer doesn't expect them),
so `december_chart.py` fills both with a single fixed value: the median
`market_index`/`quote_signal` historically observed on the Lexington → Fort
Wayne, Dry Van lane in `train_test.csv` (21 matching rows). This keeps every
row's inputs identical except date, consistent with the file's own design,
rather than introducing per-day variation the file was never meant to carry.

The scorer's own chart is saved to `outputs/scorer_results/candidate_december.png`
and reproduced in `report.docx`.
