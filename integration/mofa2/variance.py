"""
plot_variance_explained_heatmap.py
------------------------------------
Reproduces the variance-explained heatmap: factors (rows) x omics views
(columns), one panel per species/group, colored by R2. Reads
variance_explained.csv as already produced by fit_mofa_model.py's
summarize_variance_explained().

>>> Axis-swap-bug caveat <
This script trusts variance_explained.csv's 'factor' column as given. If
that file still has the known axis-swap bug (view axis mislabeled as
factor axis during reshaping -- producing only ~3 distinct "factor"
values matching the view count instead of ~15 real factors), this
heatmap will plot wrong data while still looking superficially correct
(right shape, plausible-looking colors). The script prints the distinct
factor count at the start and flags it if suspiciously low -- check that
before trusting the output.

>>> Percent-scale caveat <
mofapy2's calculate_variance_explained() output scale (0-1 fraction vs.
0-100 percent) has varied by version, and summarize_variance_explained()
in fit_mofa_model.py writes the raw value through unconverted. This
script prints the actual min/max r2 it finds rather than assuming --
set CONVERT_FRACTION_TO_PERCENT = True below if your values top out
around 0-1 instead of 0-100.

>>> Negative-R2 caveat <
R2 is computed as 1 - SS_residual/SS_total, which can come out as a tiny
negative number (e.g. -0.00003) for factors/views with ~0% real signal,
due to floating-point noise in the fit. MOFA's own R package clips these
to 0 before display; this script does the same, otherwise a near-zero
negative value shows up as a confusing "-0.0" annotation in the heatmap.

>>> Factor renumbering <
The 'factor' names coming out of fit_mofa_model.py (Factor1, Factor2, ...)
reflect the arbitrary order MOFA assigned them during training -- NOT their
importance. This script renumbers the factors it plots by total variance
explained (summed across views/groups), descending: whichever factor
explains the most variance overall becomes "Factor1" here, the second-most
becomes "Factor2", etc. This renumbering is presentation-only -- it is not
written back to variance_explained.csv, so the original MOFA factor
identities in that file are untouched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =========================================================================
# CONFIG
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

VARIANCE_EXPLAINED_CSV_PATH = BASE_DIR / "integration" / "mofa2" / "variance_explained.csv"
OUTPUT_PATH = BASE_DIR / "integration" / "mofa2" / "variance_explained_heatmap.png"

N_TOP_FACTORS = 5  # set to None to show every factor present in the file
VIEW_ORDER = ["metabolomics", "proteomics", "transcriptomics"]  # None -> pivot's default order

# Flip to True if the printed min/max at runtime shows values are 0-1
# fractions rather than 0-100 percentages.
CONVERT_FRACTION_TO_PERCENT = True

CMAP = "Purples"
FIGSIZE_PER_PANEL = (4, 5)


# =========================================================================
# Load + validate
# =========================================================================


def load_variance_explained(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    tidy = pd.read_csv(path)
    required = {"group", "factor", "view", "r2"}
    missing = required - set(tidy.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {missing}")

    n_distinct_factors = tidy["factor"].nunique()
    n_distinct_views = tidy["view"].nunique()
    print(f"Loaded {len(tidy)} rows -- {n_distinct_factors} distinct factor value(s), "
          f"groups: {sorted(tidy['group'].unique())}, views: {sorted(tidy['view'].unique(), key=str)}")
    print(f"r2 range in file: [{tidy['r2'].min():.4f}, {tidy['r2'].max():.4f}]"
          f" -- {'looks like 0-1 fractions, consider CONVERT_FRACTION_TO_PERCENT=True' if tidy['r2'].max() <= 1.5 else 'looks like percent scale already'}")

    non_string_views = [v for v in tidy["view"].unique() if not isinstance(v, str)]
    if non_string_views:
        raise ValueError(
            f"{path.name}'s 'view' column contains non-string value(s) "
            f"{sorted(non_string_views)} instead of view names like "
            f"'transcriptomics'/'proteomics'/'metabolomics'. This is the "
            f"signature of the known axis-swap bug in "
            f"summarize_variance_explained() (view axis mislabeled with "
            f"default integer column labels). Fix that function in "
            f"fit_mofa_model.py and regenerate this CSV before running this "
            f"heatmap script -- plotting it as-is would silently produce a "
            f"heatmap with factors and views swapped."
        )

    if n_distinct_factors <= n_distinct_views:
        print(f"\n[CAUTION] distinct factor count ({n_distinct_factors}) is <= distinct "
              f"view count ({n_distinct_views}) -- consistent with the known axis-swap "
              f"bug in summarize_variance_explained(). Verify this file's factor axis "
              f"is correct before trusting this heatmap.")

    # Clip tiny negative R2 values (floating-point noise for ~0%-signal
    # factor/view combos) to exactly 0, matching MOFA's own R package
    # convention -- otherwise these can render as a confusing "-0.0" label.
    n_negative = (tidy["r2"] < 0).sum()
    if n_negative:
        print(f"\n[info] clipping {n_negative} slightly negative r2 value(s) to 0 "
              f"(min was {tidy['r2'].min():.6f}) -- expected floating-point noise, "
              f"not a data error.")
    tidy = tidy.copy()
    tidy["r2"] = tidy["r2"].clip(lower=0)

    if CONVERT_FRACTION_TO_PERCENT:
        tidy["r2"] = tidy["r2"] * 100

    return tidy


def select_top_factors(tidy: pd.DataFrame, n: int | None) -> list[str]:
    """Returns original factor names (e.g. 'Factor3', 'Factor1', ...), ranked
    by total variance explained (summed across views/groups), descending."""
    total_per_factor = tidy.groupby("factor")["r2"].sum().sort_values(ascending=False)
    if n is None:
        return total_per_factor.index.tolist()
    return total_per_factor.head(n).index.tolist()


def renumber_factors_by_rank(top_factors_by_variance: list[str]) -> dict[str, str]:
    """top_factors_by_variance is already ordered highest-variance-first
    (that's what select_top_factors returns). Map that ranking directly onto
    clean sequential labels: the single highest-variance factor becomes
    'Factor1', the next becomes 'Factor2', and so on -- independent of
    whatever arbitrary name/order MOFA originally gave that factor."""
    return {orig: f"Factor{rank}" for rank, orig in enumerate(top_factors_by_variance, start=1)}


def factor_sort_key(factor_name: str):
    """'Factor12' -> 12, so ordering is numeric rather than lexicographic
    (lexicographic would put Factor10 before Factor2)."""
    digits = "".join(ch for ch in factor_name if ch.isdigit())
    return int(digits) if digits else factor_name


def build_pivot(tidy: pd.DataFrame, group: str, factors: list[str],
                 view_order: list[str] | None) -> pd.DataFrame:
    sub = tidy[(tidy["group"] == group) & (tidy["factor"].isin(factors))]
    pivot = sub.pivot_table(index="factor", columns="view", values="r2", aggfunc="mean")
    # Factor1 (highest variance explained, after renumbering) at the bottom,
    # ascending upward -- matches the target layout.
    row_order = sorted(pivot.index, key=factor_sort_key, reverse=True)
    pivot = pivot.reindex(index=row_order)
    if view_order is not None:
        pivot = pivot.reindex(columns=[v for v in view_order if v in pivot.columns])
    return pivot


# =========================================================================
# Main
# =========================================================================


def main():
    tidy = load_variance_explained(VARIANCE_EXPLAINED_CSV_PATH)

    top_factors_original = select_top_factors(tidy, N_TOP_FACTORS)
    factor_relabel = renumber_factors_by_rank(top_factors_original)
    print("\nRenumbering factors by total variance explained (highest = Factor1):")
    for orig, new in factor_relabel.items():
        print(f"  {orig} (original MOFA name) -> {new}")

    tidy = tidy[tidy["factor"].isin(top_factors_original)].copy()
    tidy["factor"] = tidy["factor"].map(factor_relabel)
    top_factors = list(factor_relabel.values())
    print(f"\nPlotting factor(s): {sorted(top_factors, key=factor_sort_key)}")

    groups = sorted(tidy["group"].unique())
    pivots = {g: build_pivot(tidy, g, top_factors, VIEW_ORDER) for g in groups}

    vmax = max(p.values.max() for p in pivots.values() if not p.empty)

    fig, axes = plt.subplots(
        1, len(groups),
        figsize=(FIGSIZE_PER_PANEL[0] * len(groups), FIGSIZE_PER_PANEL[1]),
        sharey=True,
    )
    if len(groups) == 1:
        axes = [axes]

    for i, (ax, group) in enumerate(zip(axes, groups)):
        pivot = pivots[group]
        is_last = (i == len(groups) - 1)
        sns.heatmap(
            pivot, ax=ax, cmap=CMAP, vmin=0, vmax=vmax,
            linewidths=0.5, linecolor="grey",
            cbar=is_last, cbar_kws={"label": "Var. (%)"} if is_last else None,
            annot=True, fmt=".1f", annot_kws={"size": 9},
        )
        # Default black annotation text is unreadable on the darkest cells at
        # the high end of this colormap -- switch to white above a brightness
        # threshold so numbers stay legible across the whole scale.
        for text_obj, value in zip(ax.texts, pivot.values.flatten()):
            if pd.notna(value) and vmax > 0 and (value / vmax) > 0.6:
                text_obj.set_color("white")

        # Set explicit "Factor1", "Factor2", ... row labels on every panel.
        # Because the panels share a y-axis (sharey=True), matplotlib links
        # their y Axis objects together, so labels are set on every panel and
        # only hidden visually (not cleared) on non-first panels.
        display_labels = list(pivot.index)
        ax.set_yticks([j + 0.5 for j in range(len(display_labels))])
        ax.set_yticklabels(display_labels, rotation=0)
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

        ax.set_title(group)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Factor")
        else:
            ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=90)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()