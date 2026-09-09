"""
Reproduces MOFA2's plot_weights() rank-style figure: for a given factor and
view, every feature's weight is plotted (x-axis) against its rank when
features are sorted by weight (y-axis). Only the most extreme features on
each end are labeled with their feature name; everything in between is
drawn as a thin dotted guide line rather than individual points, so the
labeled extremes stand out.

Reads weights directly from the trained model's .hdf5 file (same loader
logic/assumptions as extract_top_features.py -- see that script's docstring
for the HDF5 layout this depends on and the factor-numbering caveat).

>>> Scope <
Produces one figure per factor, with one panel per view side-by-side
(matching plot_variance_explained_heatmap.py's panel layout), for every
factor found in the model. To restrict to specific factors, set
FACTORS_TO_PLOT below.

Also writes a single tidy CSV (labeled_features.csv) listing exactly the
features labeled in the plots -- i.e. the N_LABELED_FEATURES most extreme
features per (factor, view), split evenly between the most negative
("bottom") and most positive ("top") ends. Both the raw weight and the
scaled weight (same scaling used for the plot's x-axis) are included.

>>> Why panels are NOT y-axis-shared <
Rank is relative to each view's own feature count (metabolomics might have
a few hundred features, transcriptomics tens of thousands). Sharing the
y-axis across panels (like the variance-explained heatmap does) would
visually place a smaller view's top/bottom extremes partway up the panel
instead of at the true top/bottom -- verified this failure mode directly
before removing sharey.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

# =========================================================================
# CONFIG
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

MODEL_PATH = BASE_DIR / "integration" / "mofa2" / "coffeeomics_mofa_model.hdf5"
OUTPUT_DIR = BASE_DIR / "integration" / "mofa2" / "weight_rank_plots"
LABELED_FEATURES_CSV = BASE_DIR / "integration" / "mofa2" / "labeled_features.csv"

VIEW_ORDER = ["metabolomics", "proteomics", "transcriptomics"]  # None -> file's default order

N_LABELED_FEATURES = 10  # total labeled features per panel, split evenly between top/bottom
SCALE_TO_UNIT_RANGE = True  # divide weights by max|weight| per (factor, view) -> range [-1, 1]

# Optional: restrict to specific raw MOFA factor names, e.g. ["Factor1", "Factor4"].
# None -> one figure per factor, for every factor found in the model.
FACTORS_TO_PLOT: list[str] | None = None

FIGSIZE_PER_PANEL = (4.5, 5)


# =========================================================================
# Load (same approach as extract_top_features.py)
# =========================================================================


def _decode(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def load_weights(path: Path) -> dict[str, pd.DataFrame]:
    """Returns {view_name: DataFrame (features x factors)}."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with h5py.File(path, "r") as f:
        required_top = {"views", "features", "expectations"}
        missing_top = required_top - set(f.keys())
        if missing_top or "W" not in f["expectations"]:
            print(f"\n[error] Expected top-level groups {sorted(required_top)} "
                  f"(with 'W' inside 'expectations') but found: {list(f.keys())}")
            print("[debug] Full structure:")
            f.visit(print)
            raise ValueError(
                f"{path.name} does not match the expected mofapy2 HDF5 layout "
                f"-- see the structure printed above."
            )

        views = _decode(f["views"]["views"][:])
        print(f"Found {len(views)} view(s): {views}")

        weights = {}
        for view in views:
            feature_names = _decode(f["features"][view][:])
            w = np.array(f["expectations"]["W"][view])

            if w.shape[0] == len(feature_names):
                pass
            elif w.shape[1] == len(feature_names):
                print(f"[info] W for view '{view}' came back as (factors, features) "
                      f"-- transposing.")
                w = w.T
            else:
                raise ValueError(
                    f"W shape {w.shape} for view '{view}' matches neither axis "
                    f"against {len(feature_names)} feature names."
                )

            n_factors = w.shape[1]
            factor_names = [f"Factor{i + 1}" for i in range(n_factors)]
            weights[view] = pd.DataFrame(w, index=feature_names, columns=factor_names)

    return weights


# =========================================================================
# Select + rank (shared by both the plot and the CSV, so they always agree)
# =========================================================================


def rank_and_scale(weights_series: pd.Series, scale: bool) -> pd.DataFrame:
    """Full sorted table for one (factor, view): every feature, ascending by
    raw weight, with its rank (1 = most negative) and the scaled weight used
    for the plot's x-axis (scaled == raw if scale=False)."""
    w_raw = weights_series.sort_values(ascending=True)
    if scale:
        max_abs = w_raw.abs().max()
        w_scaled = w_raw / max_abs if max_abs > 0 else w_raw.copy()
    else:
        w_scaled = w_raw

    return pd.DataFrame({
        "feature": w_raw.index,
        "rank": np.arange(1, len(w_raw) + 1),
        "weight": w_raw.values,
        "weight_scaled": w_scaled.values,
    })


def select_labeled(full_table: pd.DataFrame, n_labeled: int) -> pd.DataFrame:
    """The N_LABELED_FEATURES most extreme rows of full_table (already sorted
    ascending by weight), split evenly between the negative ("bottom") and
    positive ("top") ends -- the same features that get labeled in the plot."""
    n = len(full_table)
    n_bottom = n_labeled // 2
    n_top = n_labeled - n_bottom

    bottom = full_table.iloc[:n_bottom].copy()
    bottom["side"] = "bottom"
    top = full_table.iloc[max(n - n_top, 0):].copy()
    top["side"] = "top"

    labeled = pd.concat([bottom, top]).drop_duplicates(subset="feature")
    return labeled.sort_values("rank")


# =========================================================================
# Plot
# =========================================================================


def plot_factor_panel(ax, full_table: pd.DataFrame, labeled: pd.DataFrame, scale: bool):
    # Thin dotted guide line through every feature, sorted by weight.
    ax.plot(full_table["weight_scaled"], full_table["rank"],
             linestyle=":", color="lightgrey", linewidth=1, zorder=1)

    ax.scatter(labeled["weight_scaled"], labeled["rank"], color="black", s=15, zorder=2)

    texts = [
        ax.text(x, y, name, fontsize=8, ha="left", va="center")
        for x, y, name in zip(labeled["weight_scaled"], labeled["rank"], labeled["feature"])
    ]
    adjust_text(texts, ax=ax, arrowprops=None,
                expand_text=(1.1, 1.3), expand_points=(1.1, 1.3))

    if scale:
        ax.set_xlim(-1.15, 1.15)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Weight")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def main():
    weights = load_weights(MODEL_PATH)

    views = VIEW_ORDER if VIEW_ORDER else list(weights.keys())
    views = [v for v in views if v in weights]

    all_factors = list(next(iter(weights.values())).columns)
    factors = FACTORS_TO_PLOT if FACTORS_TO_PLOT else all_factors

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labeled_records = []

    for factor in factors:
        # NOTE: no sharey here -- rank is relative to each view's own feature
        # count, which typically differs a lot between views (e.g. metabolomics
        # vs. transcriptomics), so sharing the y-axis would visually misplace
        # a view's top/bottom extremes partway up the panel instead of at the
        # very top/bottom.
        fig, axes = plt.subplots(
            1, len(views),
            figsize=(FIGSIZE_PER_PANEL[0] * len(views), FIGSIZE_PER_PANEL[1]),
        )
        if len(views) == 1:
            axes = [axes]

        for ax, view in zip(axes, views):
            full_table = rank_and_scale(weights[view][factor], SCALE_TO_UNIT_RANGE)
            labeled = select_labeled(full_table, N_LABELED_FEATURES)

            plot_factor_panel(ax, full_table, labeled, SCALE_TO_UNIT_RANGE)
            ax.set_title(view)

            labeled_with_ids = labeled.copy()
            labeled_with_ids.insert(0, "view", view)
            labeled_with_ids.insert(0, "factor", factor)
            labeled_records.append(labeled_with_ids)

        axes[0].set_ylabel("Rank")
        fig.suptitle(factor)
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"weights_rank_{factor}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    labeled_df = pd.concat(labeled_records, ignore_index=True)
    labeled_df = labeled_df[["factor", "view", "side", "rank", "feature", "weight", "weight_scaled"]]
    LABELED_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(LABELED_FEATURES_CSV, index=False)
    print(f"\nSaved {len(labeled_df)} labeled feature row(s) to {LABELED_FEATURES_CSV}")


if __name__ == "__main__":
    main()