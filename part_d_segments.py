from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DECIMALS = 4
RANDOM_STATE = 42
N_INIT = 10
SEGMENT_K = 3  # operations runs exactly three playbooks
SEPARATION_KS = [2, 3, 4, 5]
CLUSTER_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]  # lowest to highest mean delay rate
TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def build_hub_profile(table):
    profile = table.groupby("hub_id").agg(
        shipments=("shipment_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    return profile.reset_index().sort_values("hub_id").reset_index(drop=True)


def fit_kmeans(scaled, k):
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(scaled)


def across_over_within(scaled, labels):
    """Mean distance between hubs in different clusters over the mean for hubs in the same cluster."""
    upper = np.triu_indices(len(labels), k=1)
    distance = euclidean_distances(scaled)[upper]
    same = (labels[:, None] == labels[None, :])[upper]
    return distance[~same].mean() / distance[same].mean()


def cluster_sizes(labels):
    return "-".join(str(n) for n in sorted(np.bincount(labels), reverse=True))


def main():
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(DATA / "modelling_table.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")

    # Task 1: hub profile, hubs in ascending hub_id order from here on
    profile = build_hub_profile(table)
    profile.round(DECIMALS).to_csv(OUTPUTS / "hub_profile.csv", index=False)

    # Task 2: scale the four profile columns, then cluster
    scaled = StandardScaler().fit_transform(profile[CLUSTER_COLUMNS])
    labels = fit_kmeans(scaled, SEGMENT_K).labels_

    # Task 3: separation evidence for K = 2..5 on the same scaled matrix
    separation_rows = []
    for k in SEPARATION_KS:
        k_labels = fit_kmeans(scaled, k).labels_
        separation_rows.append((k, round(across_over_within(scaled, k_labels), DECIMALS), cluster_sizes(k_labels)))
    pd.DataFrame(separation_rows, columns=["K", "across_over_within", "cluster_sizes"]).to_csv(
        OUTPUTS / "part_d_separation.csv", index=False
    )

    # Task 4: name clusters by mean hub delay rate, not by the arbitrary cluster number
    cluster_delay = profile.groupby(labels)["delay_rate"].mean().sort_values()
    names = {cluster: name for cluster, name in zip(cluster_delay.index, SEGMENT_NAMES)}
    profile["segment"] = [names[label] for label in labels]

    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
    segments = segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    segments.to_csv(OUTPUTS / "hub_segments.csv", index=False)

    segment_profile = profile.groupby("segment").agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    segment_profile.reindex(SEGMENT_NAMES).reset_index().round(DECIMALS).to_csv(
        OUTPUTS / "segment_profile.csv", index=False
    )

    implied = segments["hub_tier"].map(TIER_SEGMENT)
    segments[segments["segment"] != implied].to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)
    print(segments["segment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
