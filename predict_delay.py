import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import build_features, clean_shipments  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DECIMALS = 4
ESCALATE_ACTION = "Escalate to hub manager"
CALL_ACTION = "Proactive customer call"
NO_ACTION = "No action"


def main():
    new = pd.read_csv(DATA / "new_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    segments = pd.read_csv(OUTPUTS / "hub_segments.csv")
    model = joblib.load(MODELS / "delay_model.pkl")
    threshold = float(pd.read_csv(MODELS / "delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
    impute = pd.read_csv(MODELS / "impute_values.csv")
    medians = dict(zip(impute["column"], impute["median"]))

    clean, _ = clean_shipments(new, medians)
    features = build_features(clean, hubs)

    result = features[["shipment_id", "hub_id"]].merge(segments[["hub_id", "segment"]], on="hub_id", how="left")
    result["delay_probability"] = np.round(model.predict_proba(features[feature_order])[:, 1], DECIMALS)
    # Flag from the rounded probability so the file is self-consistent
    result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
    result["action"] = np.select(
        [(result["flagged"] == 1) & (result["segment"] == "Escalate"), result["flagged"] == 1],
        [ESCALATE_ACTION, CALL_ACTION],
        default=NO_ACTION,
    )

    OUTPUTS.mkdir(exist_ok=True)
    result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv(
        OUTPUTS / "new_shipment_actions.csv", index=False
    )
    print(len(result), int(result["flagged"].sum()), int((result["action"] == ESCALATE_ACTION).sum()))


if __name__ == "__main__":
    main()
