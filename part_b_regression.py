import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import FEATURES  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DECIMALS = 4
RANDOM_STATE = 42
TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def split_data(table):
    """70/15/15 split: train_test_split twice, random_state=42 both times."""
    X, y = table[FEATURES], table["delivery_hours"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def rounded(frame):
    return frame.round(DECIMALS)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(DATA / "modelling_table.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(table)

    # Task 2: the scaler sees the training rows only
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(float(np.abs(train_scaled.mean(axis=0)).max()), DECIMALS)),
        ("test_scaled_max_abs_mean", round(float(np.abs(test_scaled.mean(axis=0)).max()), DECIMALS)),
    ]
    pd.DataFrame(checks, columns=["metric", "value"], dtype=object).to_csv(OUTPUTS / "part_b_checks.csv", index=False)

    # Task 3: mean baseline
    baseline_pred = np.full(len(y_val), y_train.mean())
    metric_rows = [("mean_baseline", "val", regression_metrics(y_val, baseline_pred))]

    # Task 4: tree-depth diagnostic; the first depth with the lowest val_rmse wins, so ties go to the smaller depth
    depth_rows, best_depth, best_val_rmse = [], None, np.inf
    for depth in TREE_DEPTHS:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE).fit(X_train, y_train)
        train_rmse = regression_metrics(y_train, tree.predict(X_train))["RMSE"]
        val_rmse = regression_metrics(y_val, tree.predict(X_val))["RMSE"]
        depth_rows.append(("unlimited" if depth is None else depth, train_rmse, val_rmse))
        if val_rmse < best_val_rmse:
            best_depth, best_val_rmse = depth, val_rmse
    depth_table = pd.DataFrame(depth_rows, columns=["max_depth", "train_rmse", "val_rmse"], dtype=object)
    depth_table[["train_rmse", "val_rmse"]] = depth_table[["train_rmse", "val_rmse"]].astype(float).round(DECIMALS)
    depth_table.to_csv(OUTPUTS / "part_b_tree_depth.csv", index=False)

    # Task 5: four candidates scored on validation
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    }
    val_rmse_by_model = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        scores = regression_metrics(y_val, model.predict(X_val))
        metric_rows.append((name, "val", scores))
        val_rmse_by_model[name] = scores["RMSE"]

    # Task 6: gradient-boosting learning rate
    lr_rows = []
    for lr in LEARNING_RATES:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=RANDOM_STATE).fit(X_train, y_train)
        scores = regression_metrics(y_val, gb.predict(X_val))
        lr_rows.append((lr, scores["RMSE"], scores["R2"]))
    rounded(pd.DataFrame(lr_rows, columns=["learning_rate", "val_rmse", "val_r2"])).to_csv(
        OUTPUTS / "part_b_gb_learning_rate.csv", index=False
    )

    # Task 7: rank by validation RMSE, cross-validate the best two with one shared KFold, score the winner on test once
    ranking = sorted(val_rmse_by_model, key=val_rmse_by_model.get)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name in ranking[:2]:
        r2 = cross_val_score(candidates[name], X_trainval, y_trainval, cv=kfold)
        cv_rows.append((name, r2.mean(), r2.std()))
    rounded(pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"])).to_csv(OUTPUTS / "part_b_cv.csv", index=False)

    final_name = ranking[0]
    metric_rows.append((final_name, "test", regression_metrics(y_test, candidates[final_name].predict(X_test))))
    metrics = pd.DataFrame(
        [{"model": model, "split": split, **scores} for model, split, scores in metric_rows],
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
    )
    rounded(metrics).to_csv(OUTPUTS / "part_b_metrics.csv", index=False)

    print("best depth:", "unlimited" if best_depth is None else best_depth, "| ranking:", ranking, "| final:", final_name)


if __name__ == "__main__":
    main()
