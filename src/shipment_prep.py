import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
MAX_PARCEL_WEIGHT_KG = 100


def clean_shipments(raw, medians=None):
    """Clean a raw shipments table; return (clean_df, medians)."""
    df = raw.drop_duplicates().reset_index(drop=True)

    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    value_text = df["declared_value"].astype("string").str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    df["parcel_weight_kg"] = df["parcel_weight_kg"].where(df["parcel_weight_kg"] <= MAX_PARCEL_WEIGHT_KG)

    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive model inputs; never reads delivery_hours."""
    hub_columns = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_columns], on="hub_id", how="left", validate="many_to_one")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
