from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe default for headless / script runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re



# geopandas / shapely are only needed for the map functions; import lazily so
# the statistical plots still work on a machine without the geo stack.
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    _HAS_GEO = True
except Exception:  # pragma: no cover
    _HAS_GEO = False


# A small consistent colour palette for the four baselines + NN.
MODEL_COLOURS = {
    "Linear Regression": "#4e79a7",
    "K-nearest Neighbors": "#f28e2b",
}

_LSOA_SUFFIX = re.compile(r"\s+\d{3}[A-Z]$")

def _save(fig, out_path: Path, dpi: int = 150):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# A.  STATISTICAL / EVALUATION PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str | None = None,
):
    """
    Horizontal bar chart of a single metric across all models.
    `summary` is the DataFrame returned by
    calculations.average_results_across_months (index = model labels).
    """
    if metric not in summary.columns:
        raise KeyError(f"Metric '{metric}' not in summary columns {list(summary.columns)}")

    data = summary[metric].sort_values()
    colours = [MODEL_COLOURS.get(m, "#888888") for m in data.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(data.index, data.values, color=colours, edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height() / 2, f" {w:.3f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Baseline comparison — {metric}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


def plot_all_metrics_grid(summary: pd.DataFrame, out_path: Path):
    """One small bar chart per metric, in a grid — a quick visual scorecard."""
    metrics = list(summary.columns)
    n = len(metrics)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = summary[metric].sort_values()
        colours = [MODEL_COLOURS.get(m, "#888888") for m in data.index]
        ax.barh(range(len(data)), data.values, color=colours)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([m.split(" (")[0] for m in data.index], fontsize=8)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Baseline model scorecard (mean across test months)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_path)


def plot_metric_over_months(
    per_month: dict,
    metric: str,
    out_path: Path,
):
    """
    Line chart: how each model's metric changes month-to-month.
    `per_month` is {month: {model_name: {metric: value}}}.
    """
    months = sorted(per_month.keys())
    models = sorted({m for mo in per_month.values() for m in mo})

    fig, ax = plt.subplots(figsize=(12, 6))
    for model in models:
        ys = [per_month[mo].get(model, {}).get(metric, np.nan) for mo in months]
        ax.plot(months, ys, marker="o", linewidth=2, label=model.split(" (")[0])
    ax.set_xlabel("Target month")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per model across test months", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45)
    _save(fig, out_path)


def plot_crime_trend_by_force(
    crime_lsoa_month: pd.DataFrame,
    force_lookup: pd.DataFrame | None,
    out_path: Path,
):
    """
    Monthly total crime per force.
    `force_lookup` maps lsoa_code → force; if None, plots the overall total.
    """
    df = crime_lsoa_month.copy()
    fig, ax = plt.subplots(figsize=(13, 6))

    if force_lookup is not None and "force" in force_lookup.columns:
        df = df.merge(force_lookup[["lsoa_code", "force"]], on="lsoa_code", how="left")
        pivot = df.pivot_table(index="month", columns="force",
                               values="crime_count", aggfunc="sum", fill_value=0)
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
        ax.legend(title="Force", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        totals = df.groupby("month")["crime_count"].sum()
        ax.plot(totals.index, totals.values, marker="o", linewidth=2)

    ax.set_xlabel("Month")
    ax.set_ylabel("Total crime records")
    ax.set_title("Selected-crime trend over time", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45)
    _save(fig, out_path)


def plot_feature_distributions(features: pd.DataFrame, out_path: Path):
    """Histogram of each of the six engineered features (for the report's EDA)."""
    try:
        from src.calculations import FEATURE_COLUMNS
    except ImportError:
        from calculations import FEATURE_COLUMNS

    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(-1)
    for i, c in enumerate(cols):
        axes[i].hist(features[c].to_numpy(), bins=40, color="#4e79a7", edgecolor="white")
        axes[i].set_title(c, fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.3)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature distributions", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out_path)



# ═════════════════════════════════════════════════════════════════════════════
# LSOA-level regression result heatmaps (per-force grid)
# Uses the ONS 2021 GeoJSON for true LSOA polygon boundaries.
# ═════════════════════════════════════════════════════════════════════════════

def _load_lsoa_geodata(geojson_path: Path) -> "gpd.GeoDataFrame":
    """
    Load the ONS 2021 LSOA GeoJSON and assign each LSOA to one of the
    10 studied forces by extracting the LAD name from LSOA21NM and doing
    an EXACT lookup against a force -> LAD mapping.

    Why exact: LAD names overlap as substrings. 'Kingston' matches both
    Kingston upon Thames (Met) and Kingston upon Hull (Humberside);
    'Richmond' matches Richmond upon Thames and Richmondshire; 'Newcastle'
    matches Newcastle upon Tyne and Newcastle-under-Lyme. Substring matching
    grabbed those into the wrong forces and stretched the Met panel's
    bounding box from Richmondshire down to London.
    """
    if not _HAS_GEO:
        raise RuntimeError("geopandas not installed.")

    FORCE_LADS = {
        "City of London": {"City of London"},
        "Metropolitan": {
            "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
            "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
            "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
            "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea",
            "Kingston upon Thames", "Lambeth", "Lewisham", "Merton", "Newham",
            "Redbridge", "Richmond upon Thames", "Southwark", "Sutton",
            "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster",
        },
        "West Midlands": {
            "Birmingham", "Coventry", "Dudley", "Sandwell", "Solihull",
            "Walsall", "Wolverhampton",
        },
        "Merseyside": {
            "Knowsley", "Liverpool", "Sefton", "St Helens", "St. Helens", "Wirral",
        },
        "West Yorkshire": {
            "Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield",
        },
        "South Yorkshire": {"Barnsley", "Doncaster", "Rotherham", "Sheffield"},
        "Northumbria": {
            "Gateshead", "Newcastle upon Tyne", "North Tyneside",
            "Northumberland", "South Tyneside", "Sunderland",
        },
        "Leicestershire": {
            "Blaby", "Charnwood", "Harborough", "Hinckley and Bosworth",
            "Leicester", "Melton", "North West Leicestershire",
            "Oadby and Wigston",
        },
        "Nottinghamshire": {
            "Ashfield", "Bassetlaw", "Broxtowe", "Gedling", "Mansfield",
            "Newark and Sherwood", "Nottingham", "Rushcliffe",
        },
        "Humberside": {
            "East Riding of Yorkshire", "Kingston upon Hull",
            "North East Lincolnshire", "North Lincolnshire",
        },
    }

    LAD_TO_FORCE = {lad: force for force, lads in FORCE_LADS.items() for lad in lads}

    def _assign(name):
        if not isinstance(name, str):
            return None
        lad = _LSOA_SUFFIX.sub("", name).strip()
        return LAD_TO_FORCE.get(lad)

    gdf = gpd.read_file(geojson_path)
    gdf["force"] = gdf["LSOA21NM"].apply(_assign)
    return gdf[["LSOA21CD", "force", "geometry"]].rename(
        columns={"LSOA21CD": "lsoa_code"}
    )


def plot_regression_heatmaps(
    results: "pd.DataFrame",
    geojson_path: Path,
    out_path: Path,
    value_col: str = "predicted",
    title: str | None = None,
    cmap: str = "OrRd",
    figsize_per_force: tuple = (4.5, 4.5),
    cols: int = 3,
    dpi: int = 150,
    include_forces=None,    # <-- new: list of force names, or None for all
):
    """
    Draw one choropleth per force (in a grid) coloured by a column from the
    regression results table produced by `build_results_table`.

    Parameters
    ----------
    results      : DataFrame with columns ['LSOA code', value_col, ...]
                   as returned by build_results_table().
    geojson_path : path to the ONS 2021 LSOA GeoJSON.
    out_path     : where to save the PNG.
    value_col    : which column to colour by.
                   'predicted'  → predicted crime counts
                   'actual'     → ground-truth crime counts
                   'abs_error'  → absolute prediction error
                   'error'      → signed error (over/under prediction)
    title        : overall figure title (auto-generated if None).
    cmap         : matplotlib colormap.
                   'OrRd'      → white→red   (good for counts / abs error)
                   'RdBu_r'    → red→blue    (good for signed error)
    cols         : number of subplot columns in the grid.
    include_forces : optional list of force names to include (e.g. ["Metropolitan", "City of London"]).
    """
    if not _HAS_GEO:
        raise RuntimeError("geopandas not installed.")
    import pandas as pd

    # ── Load geometry + force labels ─────────────────────────────────────────
    geo = _load_lsoa_geodata(geojson_path)
    studied = geo[geo["force"].notna()].copy()

    # Filter to requested forces, if specified
    if include_forces is not None:
        studied = studied[studied["force"].isin(include_forces)]
        if studied.empty:
            raise ValueError(
                f"No LSOAs match include_forces={include_forces}. "
                f"Valid names: {sorted(geo['force'].dropna().unique())}"
        )

    # ── Merge results onto geometry ──────────────────────────────────────────
    # build_results_table uses "LSOA code" (with space); normalise to lsoa_code
    res = results.copy()
    if "LSOA code" in res.columns:
        res = res.rename(columns={"LSOA code": "lsoa_code"})

    # If multiple months, aggregate to mean per LSOA
    agg = res.groupby("lsoa_code")[value_col].mean().reset_index()

    merged = studied.merge(agg, on="lsoa_code", how="left")
    # LSOAs with no prediction stay NaN → shown as light grey

    # ── Build grid ───────────────────────────────────────────────────────────
    forces = sorted(merged["force"].dropna().unique())
    cols = min(cols, len(forces))   # don't waste columns when filtering to 1-2 forces
    rows = int(np.ceil(len(forces) / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_force[0] * cols, figsize_per_force[1] * rows),
    )
    axes = np.array(axes).reshape(-1)

    # Shared colour scale across all forces so colours are comparable
    vmin = merged[value_col].quantile(0.01)
    vmax = merged[value_col].quantile(0.99)

    for i, force in enumerate(forces):
        ax = axes[i]
        subset = merged[merged["force"] == force]

        subset.plot(
            column=value_col,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="#cccccc",
            linewidth=0.2,
            missing_kwds={"color": "#eeeeee", "label": "No data"},
            legend=False,
        )
        ax.set_title(force, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_axis_off()

    # Hide unused axes
    for j in range(len(forces), len(axes)):
        axes[j].set_visible(False)

    # Shared colourbar — dedicated axes on the right, outside the subplot grid
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])

    # Make room on the right for the colourbar, then place it there
    fig.tight_layout(rect=[0, 0, 0.92, 1])      # leave 8% on the right
    cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(value_col.replace("_", " "), fontsize=10)

    # Title
    if title is None:
        label_map = {
            "predicted":  "Predicted crime count (next month)",
            "actual":     "Actual crime count (next month)",
            "abs_error":  "Absolute prediction error",
            "error":      "Prediction error (predicted − actual)",
        }
        title = label_map.get(value_col, value_col)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure → {out_path}")