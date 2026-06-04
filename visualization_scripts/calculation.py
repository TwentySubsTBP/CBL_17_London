"""
calculation.py
==============

Loads the crime data, builds the next-month target, makes the time-based
train/test split, and provides the two baseline models (Linear Regression and
KNN Regression) plus helpers used by visualization.py.

Design:
  * All the data prep runs at import time (cheap, just loads one parquet) so
    that `df`, `test_mask`, `y_test`, `X_train`, ... are available to any file
    that imports this module.
  * The actual model runs and the printed report only happen when you run this
    file directly (python calculation.py), guarded by __main__.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Configuration 
DATA_PATH = Path(r"scripts\crime_data.parquet")
LSOA_COL = "LSOA code"
MONTH_COL = "Month"

# Load data 
df = pd.read_parquet(DATA_PATH)
df[MONTH_COL] = pd.to_datetime(df[MONTH_COL])
df = df.sort_values([LSOA_COL, MONTH_COL]).reset_index(drop=True)

# Build target (next month's crime count) 
df["next_month"] = df.groupby(LSOA_COL)[MONTH_COL].shift(-1)
df["crime_next_month"] = df.groupby(LSOA_COL)["crime_count"].shift(-1)

# Keep only rows where next month is truly consecutive
df["expected_next"] = df[MONTH_COL] + pd.DateOffset(months=1)
df = df[df["next_month"] == df["expected_next"]].copy()
df["target_month"] = df["next_month"]
df = df.dropna(subset=["crime_next_month"])

#Features 
candidate_features = [
    "crime_count",
    "crime_1m_ago",
    "crime_3m_ago",
    "crime_6m_ago",
    "yearly_avg",
    "neighbor_crime_count",
    "neighbor_1m_ago",
    "neighbor_3m_ago",
    "neighbor_6m_ago",
]
feature_cols = [c for c in candidate_features if c in df.columns]

X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
y = df["crime_next_month"].to_numpy(dtype=np.float32)

# Time-based split 
target_months = pd.to_datetime(df["target_month"])
train_mask = target_months < pd.Timestamp("2025-07-01")
test_mask = target_months >= pd.Timestamp("2025-07-01")

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Standardise 
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std = np.where(std == 0, 1.0, std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Metrics helper
def evaluate(name, y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"{name:30s} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    return mae, rmse


# Baseline models — each returns its prediction array
def calculate_linear_regression():
    model = LinearRegression()
    model.fit(X_train, np.log1p(y_train))

    raw_log_preds = model.predict(X_test)
    # Clip in log space at the 99th percentile of training counts to stop
    # linear extrapolation producing absurd values for outlier LSOAs.
    cap_log = np.log1p(np.percentile(y_train, 99)) + 0.5
    log_preds_clipped = np.clip(raw_log_preds, 0, cap_log)
    preds = np.expm1(log_preds_clipped)
    preds = np.maximum(preds, 0)

    evaluate("Linear Regression", y_test, preds)
    return preds


def calculate_knn_regression(k=10, weights="uniform"):
    model = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)
    model.fit(X_train, np.log1p(y_train))

    preds = np.expm1(model.predict(X_test))
    preds = np.maximum(preds, 0)

    evaluate(f"KNN (k={k}, {weights})", y_test, preds)
    return preds


# Accessors for other modules (e.g. visualization.py)
def get_data_split():
    """Return (df, test_mask, y_test) for building results tables."""
    return df, test_mask, y_test



# Turn a bare prediction array into a tidy, visualization-ready table
def build_results_table(df, test_mask, y_test, predictions, model_name,
                        lsoa_col="LSOA code", month_col="target_month"):
    """
    Re-attach LSOA code + target month to a prediction array.

    The same boolean `test_mask` that produced X_test / y_test also selects the
    matching rows of `df`, so we reuse it to recover identifiers.

    Returns a DataFrame with columns:
        LSOA code, target_month, model, actual, predicted, error, abs_error
    """
    df_test = df[test_mask].copy().reset_index(drop=True)

    if len(df_test) != len(predictions):
        raise ValueError(
            f"Length mismatch: df_test has {len(df_test)} rows but predictions "
            f"has {len(predictions)}. Ensure test_mask, y_test and predictions "
            f"come from the same split."
        )

    results = pd.DataFrame({
        "LSOA code": df_test[lsoa_col].values,
        "target_month": pd.to_datetime(df_test[month_col].values),
        "model": model_name,
        "actual": np.asarray(y_test, dtype=float),
        "predicted": np.asarray(predictions, dtype=float),
    })
    results["error"] = results["predicted"] - results["actual"]
    results["abs_error"] = results["error"].abs()
    return results


def filter_results_by_month(
    results: pd.DataFrame,
    months=None,
    start=None,
    end=None,
    month_col: str = "target_month",
) -> pd.DataFrame:
    """
    Filter the results table by target month.

    Use one of:
      - months="2025-07"                 single month
      - months=["2025-07", "2025-08"]    multiple specific months
      - start="2025-07"                  open-ended from (inclusive)
      - end="2025-09"                    open-ended to (inclusive)
      - start + end                      range (inclusive both sides)

    Strings accept either "YYYY-MM" or "YYYY-MM-DD" (day is normalised to the 1st).
    Returns a filtered copy; raises ValueError if no rows match.
    """
    if months is None and start is None and end is None:
        return results  # nothing to filter

    df = results.copy()
    months_in_df = pd.to_datetime(df[month_col])

    def _to_ts(s):
        s = str(s)
        return pd.to_datetime(f"{s}-01" if len(s) == 7 else s)

    if months is not None:
        if isinstance(months, str):
            months = [months]
        wanted = pd.to_datetime([
            f"{m}-01" if len(str(m)) == 7 else m for m in months
        ])
        mask = months_in_df.isin(wanted)
    else:
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= months_in_df >= _to_ts(start)
        if end is not None:
            mask &= months_in_df <= _to_ts(end)

    filtered = df[mask].copy()

    if filtered.empty:
        available = sorted(months_in_df.dt.strftime("%Y-%m").unique())
        raise ValueError(
            f"No rows match the filter. Available months: {available}"
        )

    n_kept = filtered[month_col].nunique()
    print(f"[filter] kept {len(filtered):,} rows across {n_kept} month(s)")
    return filtered

def remap_results(
    results: pd.DataFrame,
    lookup_path: str | Path = "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv",
    lsoa_col: str = "LSOA code",
    month_col: str = "target_month",
    value_cols: tuple = ("actual", "predicted"),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return a new results DataFrame keyed on 2021 LSOA codes.

    Drop-in replacement for the original results table — same column names,
    so plot_regression_heatmaps() works without changes.
    """
    lookup = pd.read_csv(lookup_path, usecols=["LSOA11CD", "LSOA21CD", "CHGIND"])

    # How many 2021 codes does each 2011 code map to? (S/X cases produce >1)
    expansion = (lookup.groupby("LSOA11CD")["LSOA21CD"]
                       .count()
                       .rename("n_targets"))
    lookup = lookup.merge(expansion, on="LSOA11CD")

    df = results.copy()

    # Join on LSOA code — multiple rows in lookup means row is replicated
    df = df.merge(
        lookup[["LSOA11CD", "LSOA21CD", "CHGIND", "n_targets"]],
        left_on=lsoa_col, right_on="LSOA11CD", how="left",
    )

    # Rows with no lookup match: code is presumed already a 2021 code
    no_match = df["LSOA21CD"].isna()
    df.loc[no_match, "LSOA21CD"] = df.loc[no_match, lsoa_col]
    df.loc[no_match, "n_targets"] = 1

    # Equal-divide value columns for splits (n_targets > 1)
    for c in value_cols:
        if c in df.columns:
            df[c] = df[c] / df["n_targets"]

    # Aggregate to one row per 2021 LSOA × month × model (sums for merges)
    group_cols = ["LSOA21CD"]
    if month_col in df.columns:
        group_cols.append(month_col)
    if "model" in df.columns:
        group_cols.append("model")

    agg_cols = [c for c in value_cols if c in df.columns]
    out = df.groupby(group_cols, as_index=False)[agg_cols].sum()

    # Recompute errors from the new totals
    if {"actual", "predicted"}.issubset(out.columns):
        out["error"] = out["predicted"] - out["actual"]
        out["abs_error"] = out["error"].abs()

    # Rename back so it's a drop-in for the original
    out = out.rename(columns={"LSOA21CD": lsoa_col})

    if verbose:
        n_in = len(results)
        n_out = len(out)
        n_unique_in = results[lsoa_col].nunique()
        n_unique_out = out[lsoa_col].nunique()
        print(f"[remap] rows  : {n_in:,} -> {n_out:,}")
        print(f"[remap] LSOAs : {n_unique_in:,} -> {n_unique_out:,} (all 2021)")

    return out



# Only runs when you execute this file directly
if __name__ == "__main__":
    from pathlib import Path
    from visualization_scripts.visualization import plot_regression_heatmaps

    GEOJSON = Path("LSOA_(2021)_EW_BSC_V4_to_Rural_Urban_Classification.geojson")
    LOOKUP  = Path("LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv");

    # 1. Load data split (already done at import time, just fetch refs)
    df, test_mask, y_test = get_data_split()

    # 2. Run the model
    lr_preds = calculate_linear_regression()

    # 3. Build the results table (2011-coded LSOAs still in here)
    results = build_results_table(df, test_mask, y_test, lr_preds, "Linear Regression")

    # 4. Remap 2011 -> 2021 LSOA codes so every code matches the GeoJSON
    results = remap_results(results, lookup_path=LOOKUP)

    # 5. Plot — all four heatmaps use the same (now-clean) results table

    # Predicted crime heatmap
    plot_regression_heatmaps(results, GEOJSON,
        out_path=Path("figures/lr_predicted.png"),
        value_col="predicted", cmap="OrRd",
        title="Linear Regression — Predicted crime (next month)")

    # Ground truth heatmap
    plot_regression_heatmaps(results, GEOJSON,
        out_path=Path("figures/lr_actual.png"),
        value_col="actual", cmap="OrRd",
        title="Ground truth — Actual crime (next month)")

    # Absolute error heatmap
    plot_regression_heatmaps(results, GEOJSON,
        out_path=Path("figures/lr_abs_error.png"),
        value_col="abs_error", cmap="OrRd",
        title="Linear Regression — Absolute error per LSOA")

    # Signed error — red=overpredicted, blue=underpredicted
    plot_regression_heatmaps(results, GEOJSON,
        out_path=Path("figures/lr_error.png"),
        value_col="error", cmap="RdBu_r",
        title="Linear Regression — Prediction error (predicted − actual)")
