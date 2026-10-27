import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import FEATURES, NOISE_COLUMNS  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DECIMALS = 4
RANDOM_STATE = 42
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_KS = [5, 7, 10, 12, 17]
CHOSEN_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RECALL_TARGET = 0.80
CONFUSION_COLUMNS = ["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
# Tuning candidates that tie on validation F1 are resolved in this order
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]


def split_data(table):
    """Stratified 70/15/15 split: two train_test_split calls, random_state=42, stratified both times."""
    X, y = table[FEATURES], table["is_delayed"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def cell_metrics(y_true, y_pred):
    """Confusion-matrix cells plus the four scores derived from them."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def format_value(value):
    if value is None or isinstance(value, (str, bool, np.bool_)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), DECIMALS)


def write_csv(rows, columns, path):
    frame = pd.DataFrame([[format_value(v) for v in row] for row in rows], columns=columns, dtype=object)
    frame.to_csv(path, index=False)
    return frame


def params_text(params):
    return str({key: int(value) for key, value in sorted(params.items())})


def main():
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(DATA / "modelling_table.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(table)

    # Task 2: blind baseline and three classifiers on validation at the default 0.5 rule
    model_rows = []

    def add_model_row(name, y_pred):
        m = cell_metrics(y_val, y_pred)
        model_rows.append([name] + [m[c] for c in CONFUSION_COLUMNS])
        return m

    add_model_row("blind_all_on_time", np.zeros(len(y_val), dtype=int))
    logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    add_model_row("logistic_regression", logit.predict(X_val))

    best_depth, best_f1 = None, -1.0
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=RANDOM_STATE).fit(X_train, y_train)
        f1 = round(cell_metrics(y_val, tree.predict(X_val))["f1"], DECIMALS)
        if f1 > best_f1:
            best_depth, best_f1 = depth, f1
    tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    add_model_row(f"decision_tree_depth_{best_depth}", tree.predict(X_val))

    forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)
    add_model_row("random_forest_default", forest.predict(X_val))
    write_csv(model_rows, ["model"] + CONFUSION_COLUMNS, OUTPUTS / "part_c_models.csv")

    # Task 3: which columns carry signal; selector and forest see the training rows only
    all_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
    importance = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train).feature_importances_
    scores = pd.DataFrame({"feature": FEATURES, "f_score": all_scores, "rf_importance": importance})
    scores["f_rank"] = scores["f_score"].rank(ascending=False, method="min").astype(int)
    scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
    scores = scores.sort_values("f_rank", kind="stable")
    write_csv(
        scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].values.tolist(),
        ["feature", "f_score", "f_rank", "rf_importance", "is_noise"],
        OUTPUTS / "part_c_feature_scores.csv",
    )

    selectk_rows = []
    for k in SELECT_KS:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]).fit(X_train, y_train)
        selectk_rows.append([k, cell_metrics(y_val, pipe.predict(X_val))["f1"]])
    write_csv(selectk_rows, ["k", "val_f1"], OUTPUTS / "part_c_selectk.csv")

    # Task 4: tuning on the 7 selected columns; the test rows are not touched
    selector = SelectKBest(f_classif, k=CHOSEN_K).fit(X_train, y_train)
    X_train_sel, X_val_sel = selector.transform(X_train), selector.transform(X_val)

    grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID, cv=5).fit(X_train_sel, y_train)
    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID, n_iter=9, cv=5, random_state=RANDOM_STATE
    ).fit(X_train_sel, y_train)
    default = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_train_sel, y_train)

    tuning_rows, candidate_params = [], {"default_forest_same_7_features": {}}
    for name, search in [("grid_search", grid), ("randomized_search", randomized)]:
        candidates = len(search.cv_results_["params"])
        folds = int(search.n_splits_)
        candidate_params[name] = {key: int(value) for key, value in search.best_params_.items()}
        tuning_rows.append([
            name, candidates, folds, candidates * folds, params_text(search.best_params_),
            search.best_score_, cell_metrics(y_val, search.best_estimator_.predict(X_val_sel))["f1"],
        ])
    tuning_rows.append([
        "default_forest_same_7_features", 1, 0, 1, "defaults", None, cell_metrics(y_val, default.predict(X_val_sel))["f1"],
    ])
    tuning = write_csv(
        tuning_rows,
        ["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"],
        OUTPUTS / "part_c_tuning.csv",
    )

    # Task 5: highest validation F1 as written in the table; ties follow TIE_ORDER
    winner = max(tuning["search"], key=lambda s: (tuning.set_index("search").loc[s, "val_f1"], -TIE_ORDER.index(s)))
    params = candidate_params[winner]
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=CHOSEN_K)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, **params)),
    ]).fit(X_train, y_train)

    # Task 6: alert threshold from validation probabilities
    val_proba = final_model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for t in THRESHOLDS:
        m = cell_metrics(y_val, (val_proba >= t).astype(int))
        threshold_rows.append([t] + [m[c] for c in CONFUSION_COLUMNS])
    thresholds = write_csv(threshold_rows, ["threshold"] + CONFUSION_COLUMNS, OUTPUTS / "part_c_thresholds.csv")
    meets_target = thresholds[thresholds["recall"] >= RECALL_TARGET]
    threshold = float(meets_target["threshold"].max()) if len(meets_target) else 0.1

    # Task 7: the single test evaluation at the chosen threshold
    test_proba = final_model.predict_proba(X_test)[:, 1]
    m = cell_metrics(y_test, (test_proba >= threshold).astype(int))
    write_csv([[threshold] + [m[c] for c in CONFUSION_COLUMNS]], ["threshold"] + CONFUSION_COLUMNS, OUTPUTS / "part_c_final_test.csv")
    write_csv(
        [[
            CHOSEN_K, params.get("n_estimators", 100), params.get("max_depth", "unlimited"),
            params.get("min_samples_leaf", 1), threshold, winner,
        ]],
        ["selector_k", "n_estimators", "max_depth", "min_samples_leaf", "threshold", "chosen_from"],
        OUTPUTS / "part_c_final_config.csv",
    )
    print("chosen:", winner, params, "threshold:", threshold)


if __name__ == "__main__":
    main()
