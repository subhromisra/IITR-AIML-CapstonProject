import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from part_c_classification import RANDOM_STATE, split_data  # noqa: E402
from shipment_prep import FEATURES, clean_shipments  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DECIMALS = 4


def build_final_pipeline(config):
    """Rebuild the Part C final pipeline from part_c_final_config.csv."""
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    return Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=int(config["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(config["min_samples_leaf"]),
        )),
    ])


def main():
    MODELS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    config = pd.read_csv(OUTPUTS / "part_c_final_config.csv").iloc[0]
    threshold = float(config["threshold"])

    table = pd.read_csv(DATA / "modelling_table.csv")
    X_train, _, X_test, y_train, _, _ = split_data(table)
    model = build_final_pipeline(config).fit(X_train, y_train)

    # The fill-in medians come from the same historical table the model was trained on
    _, medians = clean_shipments(pd.read_csv(DATA / "raw_shipments.csv"))

    joblib.dump(model, MODELS / "delay_model.pkl")
    pd.DataFrame({"threshold": [threshold]}).to_csv(MODELS / "delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)
    pd.DataFrame(
        {"column": list(medians), "median": [round(value, DECIMALS) for value in medians.values()]}
    ).to_csv(MODELS / "impute_values.csv", index=False)

    reloaded = joblib.load(MODELS / "delay_model.pkl")
    in_memory_proba = model.predict_proba(X_test)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
    checks = pd.DataFrame(
        [
            ("test_rows_scored", len(X_test)),
            ("reload_max_abs_probability_diff", round(float(np.abs(in_memory_proba - reloaded_proba).max()), DECIMALS)),
            ("threshold", threshold),
        ],
        columns=["metric", "value"],
        dtype=object,
    )
    checks.to_csv(OUTPUTS / "part_d_reload_check.csv", index=False)
    print(len(X_test), float(np.abs(in_memory_proba - reloaded_proba).max()), threshold)


if __name__ == "__main__":
    main()
