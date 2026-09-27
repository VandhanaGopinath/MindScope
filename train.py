"""
MindScope - training script
Trains one XGBoost regressor per Big Five trait (E, N, A, C, O) that
predicts the FULL 10-item trait score from only the FIRST 5 items of that
trait (a 25-question short form instead of the full 50-question test).

Two fixes over the original notebook:
1. Reverse-scoring: negatively-worded items (e.g. "I don't talk a lot")
   are now flipped (6 - raw) using the standard IPIP-50 key before scoring.
   The original notebook averaged raw responses with no reverse-scoring,
   which biases every trait score.
2. Non-circular target: the original notebook fed the model the SAME 10
   items it was trying to average (X = all 50 items, y = mean of 10 of
   them), so "prediction" was really just re-deriving an average -> R2
   was ~0.997 by construction, not because the model learned anything.
   Here the model only sees 5 of the 10 items per trait and has to
   estimate what the full 10-item average would have been -> a genuine
   (and still strong) predictive task: a shorter quiz standing in for
   the full one.

Run:
    python train.py
Produces:
    models/model_<TRAIT>.joblib   (one per trait)
    models/feature_columns.joblib
    models/metrics.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

TRAITS = ["E", "N", "A", "C", "O"]
ITEM_COLS = [f"{t}{i}" for t in TRAITS for i in range(1, 11)]
# Short-form input: only the first 5 items of each trait are used as features
SHORT_FORM_COLS = [f"{t}{i}" for t in TRAITS for i in range(1, 6)]

# Standard IPIP-50 scoring key: True = item is reverse-scored (6 - raw)
REVERSE_KEY = {
    "E1": False, "E2": True,  "E3": False, "E4": True,  "E5": False,
    "E6": True,  "E7": False, "E8": True,  "E9": False, "E10": True,
    "N1": False, "N2": True,  "N3": False, "N4": True,  "N5": False,
    "N6": False, "N7": False, "N8": False, "N9": False, "N10": False,
    "A1": True,  "A2": False, "A3": True,  "A4": False, "A5": True,
    "A6": False, "A7": True,  "A8": False, "A9": False, "A10": False,
    "C1": False, "C2": True,  "C3": False, "C4": True,  "C5": False,
    "C6": True,  "C7": False, "C8": True,  "C9": False, "C10": False,
    "O1": False, "O2": True,  "O3": False, "O4": True,  "O5": False,
    "O6": True,  "O7": False, "O8": False, "O9": False, "O10": False,
}


def load_and_clean(path="data.csv"):
    data = pd.read_csv(path, sep="\t")
    # 0 = missed/skipped per the dataset README -> drop those rows
    data = data[(data[ITEM_COLS] != 0).all(axis=1)].reset_index(drop=True)
    return data


def apply_reverse_scoring(data):
    scored = data[ITEM_COLS].copy()
    for col, reverse in REVERSE_KEY.items():
        if reverse:
            scored[col] = 6 - scored[col]
    return scored


def build_targets(scored):
    targets = pd.DataFrame()
    for t in TRAITS:
        cols = [f"{t}{i}" for i in range(1, 11)]
        targets[t] = scored[cols].mean(axis=1)
    return targets


def main():
    data = load_and_clean()
    print(f"Rows after dropping missing/skipped answers: {len(data)}")

    scored = apply_reverse_scoring(data)
    y = build_targets(scored)  # full 10-item average per trait (the "true" score)
    X = scored[SHORT_FORM_COLS]  # model input: only 5 of the 10 items per trait

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = {}
    for t in TRAITS:
        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train[t])
        pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test[t], pred)))
        r2 = float(r2_score(y_test[t], pred))
        metrics[t] = {"rmse": rmse, "r2": r2}
        print(f"{t}: RMSE={rmse:.4f}  R2={r2:.4f}")

        joblib.dump(model, f"models/model_{t}.joblib")

    joblib.dump(SHORT_FORM_COLS, "models/feature_columns.joblib")
    joblib.dump(REVERSE_KEY, "models/reverse_key.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved models + metrics to ./models/")


if __name__ == "__main__":
    main()
