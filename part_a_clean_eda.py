import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import FEATURES, MAX_PARCEL_WEIGHT_KG, build_features, clean_shipments  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DECIMALS = 4
DISTANCE_EDGES = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]


def build_checks(raw, table, medians):
    """Metrics for part_a_checks.csv, all measured from the data."""
    deduped = raw.drop_duplicates()
    service_after = deduped["service_type"].str.strip().str.title()
    declared = table["declared_value_inr"]
    log_declared = table["log_declared_value"]

    hours = table["delivery_hours"]
    q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

    return [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(deduped)),
        ("rows_after_dedup", len(deduped)),
        ("missing_distance_km_before", deduped["distance_km"].isna().sum()),
        ("missing_parcel_weight_before", deduped["parcel_weight_kg"].isna().sum()),
        ("sentinel_weight_rows", (deduped["parcel_weight_kg"] > MAX_PARCEL_WEIGHT_KG).sum()),
        ("missing_declared_value_before", deduped["declared_value"].isna().sum()),
        ("missing_payment_mode_before", deduped["payment_mode"].isna().sum()),
        ("service_type_spellings_before", deduped["service_type"].nunique()),
        ("service_type_spellings_after", service_after.nunique()),
        ("declared_value_mean_inr", declared.mean()),
        ("declared_value_median_inr", declared.median()),
        ("log_declared_value_mean", log_declared.mean()),
        ("log_declared_value_median", log_declared.median()),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(table)),
        ("nulls_in_features_and_target", table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        ("delayed_count", table["is_delayed"].sum()),
        ("delay_rate", table["is_delayed"].mean()),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower),
        ("hours_upper_bound", upper),
        ("hours_outlier_count", ((hours < lower) | (hours > upper)).sum()),
    ]


def format_value(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), DECIMALS)


def build_delay_by_group(table):
    """Shipments, delayed count, delay rate and mean hours per service type, hub tier and distance band."""
    table = table.assign(
        distance_band=pd.cut(
            table["distance_km"], bins=DISTANCE_EDGES, labels=DISTANCE_LABELS, right=True, include_lowest=True
        )
    )
    frames = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        grouped = table.groupby(dimension, observed=True).agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        )
        grouped = grouped.reset_index().rename(columns={dimension: "group"})
        grouped["group"] = grouped["group"].astype(str)
        grouped.insert(0, "dimension", dimension)
        frames.append(grouped)
    result = pd.concat(frames, ignore_index=True)
    result[["delay_rate", "mean_hours"]] = result[["delay_rate", "mean_hours"]].round(DECIMALS)
    return result


def main():
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")

    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)

    table.to_csv(DATA / "modelling_table.csv", index=False)

    OUTPUTS.mkdir(exist_ok=True)
    checks = pd.DataFrame(
        [(metric, format_value(value)) for metric, value in build_checks(raw, table, medians)],
        columns=["metric", "value"],
        dtype=object,
    )
    checks.to_csv(OUTPUTS / "part_a_checks.csv", index=False)
    build_delay_by_group(table).to_csv(OUTPUTS / "part_a_delay_by_group.csv", index=False)
    table.select_dtypes(include="number").corr().round(DECIMALS).to_csv(OUTPUTS / "part_a_correlations.csv")
    print(len(table), int(table["is_delayed"].sum()))


if __name__ == "__main__":
    main()
