"""
plot_factor_scatter_pairs.py
------------------------------
For every pairwise combination of the top 5 MOFA factors (by summed
variance explained), generates a scatter plot (factor_i vs factor_j),
faceted by species, for each of three condition variables: Temperature,
CO2_ppm, Water_Stress. Colors come from the project's own color_maps.py.

5 factors -> C(5,2) = 10 pairs x 3 variables = 30 plots total.

Mirrors MOFA2's plotFactorScatter convention: dashed reference lines at
x=0 and y=0 on each panel.

This script is self-contained (does not import from the beeswarm script)
so it isn't coupled to whatever filename that script ends up saved under
locally -- the small amount of shared logic (color map loading, top-factor
selection, cross-layer metadata merge, Windows-lock-safe savefig) is
duplicated here rather than imported.

Inputs:
  - variance_explained.csv   (output of fit_mofa_model.py)
  - factor_scores_long.csv   (output of extract_factor_scores.py)
  - contrast_groups_{transcriptomics,proteomics,metabolomics}.json
  - color_maps.py            (imported directly from BASE_DIR)

>>> Dependency caveat <<<
Same axis-swap-bug caveat as the beeswarm script: if
variance_explained.csv still has only ~3 distinct "factor" values
(matching view count, not real factor count), the top-5 ranking here is
unreliable. Use MANUAL_TOP_FACTORS below to bypass it if that's still
unresolved.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =========================================================================
# CONFIG
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

VARIANCE_EXPLAINED_CSV_PATH = BASE_DIR / "integration" / "mofa2" / "variance_explained.csv"
FACTOR_SCORES_LONG_PATH = BASE_DIR / "integration" / "mofa2" / "factor_scores_long.csv"
COLOR_MAPS_PATH = BASE_DIR / "color_maps.py"

CONTRAST_GROUPS_TRANSCRIPTOMICS_PATH = BASE_DIR / "transcriptomics" / "contrast_groups_transcriptomics.json"
CONTRAST_GROUPS_PROTEOMICS_PATH = BASE_DIR / "proteomics" / "contrast_groups_proteomics.json"
CONTRAST_GROUPS_METABOLOMICS_PATH = BASE_DIR / "metabolomics" / "contrast_groups_metabolomics.json"

OUTPUT_DIR = BASE_DIR / "integration" / "mofa2" / "factor_scatter_plots"

N_TOP_FACTORS = 5

# Set to e.g. ["Factor3", "Factor7", ...] to bypass the variance_explained.csv
# ranking entirely (useful while the axis-swap bug is unresolved).
MANUAL_TOP_FACTORS: list[str] | None = None

CONDITION_VARIABLES = ["Temperature", "CO2_ppm", "Water_Stress"]

DOT_SIZE = 30       # matplotlib scatter 's' units (points^2), not the same scale as beeswarm's DOT_SIZE
FIGSIZE = (10, 5)

# =========================================================================
# Shared loaders (duplicated from the beeswarm script -- see module docstring)
# =========================================================================


def load_color_maps(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"color_maps.py not found at {path}")
    spec = importlib.util.spec_from_file_location("color_maps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "COLOR_MAPS"):
        raise AttributeError(f"{path} does not define COLOR_MAPS")
    return module.COLOR_MAPS


def get_top_factors(variance_csv: Path, n: int) -> list[str]:
    if MANUAL_TOP_FACTORS is not None:
        print(f"Using MANUAL_TOP_FACTORS override: {MANUAL_TOP_FACTORS}")
        return MANUAL_TOP_FACTORS

    if not variance_csv.exists():
        raise FileNotFoundError(f"{variance_csv} not found")
    tidy = pd.read_csv(variance_csv)
    n_distinct_factors = tidy["factor"].nunique()
    total_per_factor = tidy.groupby("factor")["r2"].sum().sort_values(ascending=False)
    top = total_per_factor.head(n).index.tolist()
    print("Top factors by summed variance explained:")
    print(total_per_factor.head(n).to_string())

    if n_distinct_factors <= 3:
        print(f"\n[CAUTION] {variance_csv.name} contains only {n_distinct_factors} "
              f"distinct 'factor' values -- consistent with the known axis-swap "
              f"bug (view axis mislabeled as factor axis). This ranking is "
              f"almost certainly wrong. Set MANUAL_TOP_FACTORS above to bypass.")
    elif len(top) < n:
        print(f"\n[warn] only {len(top)} distinct factor(s) found in "
              f"{variance_csv.name}, fewer than the requested {n}.")
    return top


def invert_contrast_groups(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        contrast_groups = json.load(f)

    records = []
    for condition_str, sample_ids in contrast_groups.items():
        parts = [p.strip() for p in condition_str.split("|")]
        if len(parts) != 4:
            print(f"  [warn] unexpected condition string format in {path.name}, "
                  f"skipping: {condition_str!r}", file=sys.stderr)
            continue
        species, temperature, co2, water_stress = parts
        for sid in sample_ids:
            records.append({
                "sample": sid,
                "species": species,
                "Temperature": temperature,
                "CO2_ppm": co2,
                "Water_Stress": water_stress,
            })
    return pd.DataFrame(records)


def build_merged_sample_metadata() -> pd.DataFrame:
    layer_frames = {}
    for layer_name, path in [
        ("transcriptomics", CONTRAST_GROUPS_TRANSCRIPTOMICS_PATH),
        ("proteomics", CONTRAST_GROUPS_PROTEOMICS_PATH),
        ("metabolomics", CONTRAST_GROUPS_METABOLOMICS_PATH),
    ]:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        layer_frames[layer_name] = invert_contrast_groups(path).set_index("sample")

    if not layer_frames:
        raise FileNotFoundError("No contrast_groups files could be loaded")

    all_samples = sorted(set().union(*[df.index for df in layer_frames.values()]))
    cond_cols = ["species", "Temperature", "CO2_ppm", "Water_Stress"]

    merged_records = []
    conflicts = []
    for sample_id in all_samples:
        values_by_layer = {
            layer: df.loc[sample_id, cond_cols].to_dict()
            for layer, df in layer_frames.items() if sample_id in df.index
        }
        distinct_value_sets = {tuple(v.values()) for v in values_by_layer.values()}
        if len(distinct_value_sets) > 1:
            conflicts.append((sample_id, values_by_layer))
        chosen = next(iter(values_by_layer.values()))
        merged_records.append({"sample": sample_id, **chosen})

    if conflicts:
        print(f"\n[CONFLICT] {len(conflicts)} sample ID(s) have DIFFERENT conditions "
              f"across layers -- values used are from the first layer found; "
              f"verify these manually before trusting their plotted color:")
        for sample_id, values_by_layer in conflicts:
            print(f"  {sample_id}: {values_by_layer}")

    return pd.DataFrame(merged_records)


def savefig_with_retry(fig, path: Path, attempts: int = 4, delay: float = 1.5, **kwargs) -> None:
    """
    See beeswarm script for rationale: retries on OSError since a save that
    fails only after several identical prior saves succeeded is almost
    always a transient Windows file lock (OneDrive sync, Defender
    real-time scanning), not a code bug.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError as e:
            last_err = e
            print(f"  [warn] savefig attempt {attempt}/{attempts} failed for "
                  f"{path.name}: {e}")
            time.sleep(delay)
    raise OSError(
        f"Could not save {path} after {attempts} attempts ({last_err}). "
        f"Likely a Windows file lock (OneDrive sync, Defender real-time "
        f"scanning, or a preview handler). Try pausing OneDrive sync / "
        f"Defender real-time protection for this folder, or set OUTPUT_DIR "
        f"to a path outside OneDrive-synced Desktop, then rerun."
    ) from last_err


# =========================================================================
# Plotting
# =========================================================================


def plot_pair(wide_df: pd.DataFrame, factor_x: str, factor_y: str, variable: str,
              color_map: dict, output_path: Path) -> None:
    species_list = sorted(wide_df["group"].unique())
    fig, axes = plt.subplots(1, len(species_list), figsize=FIGSIZE, sharex=True, sharey=True)
    if len(species_list) == 1:
        axes = [axes]

    extended_palette = dict(color_map)

    for ax, species in zip(axes, species_list):
        sub = wide_df[wide_df["group"] == species]

        missing_from_map = sorted(set(sub[variable].dropna().unique()) - set(color_map.keys()))
        if missing_from_map:
            print(f"  [warn] {factor_x}_vs_{factor_y}/{variable}/{species}: value(s) "
                  f"not in COLOR_MAPS, plotted grey: {missing_from_map}")
            for level in missing_from_map:
                extended_palette.setdefault(level, "#999999")

        colors = sub[variable].map(extended_palette).fillna("#999999")

        ax.scatter(sub[factor_x], sub[factor_y], c=colors, s=DOT_SIZE,
                   edgecolors="black", linewidths=0.4)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.7)
        ax.set_title(species)
        ax.set_xlabel(factor_x)

    axes[0].set_ylabel(factor_y)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=level,
                          markerfacecolor=color, markeredgecolor="black", markersize=8)
               for level, color in extended_palette.items()]
    fig.legend(handles=handles, title=variable, loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{factor_x} vs {factor_y}, colored by {variable}")
    fig.tight_layout()
    savefig_with_retry(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# =========================================================================
# Main
# =========================================================================


def main():
    color_maps = load_color_maps(COLOR_MAPS_PATH)
    for variable in CONDITION_VARIABLES:
        if variable not in color_maps:
            raise KeyError(f"COLOR_MAPS in {COLOR_MAPS_PATH} has no key {variable!r}; "
                            f"available keys: {list(color_maps.keys())}")

    top_factors = get_top_factors(VARIANCE_EXPLAINED_CSV_PATH, N_TOP_FACTORS)
    factor_pairs = list(itertools.combinations(top_factors, 2))
    print(f"\n{len(top_factors)} factors -> {len(factor_pairs)} pairwise combination(s)")

    scores = pd.read_csv(FACTOR_SCORES_LONG_PATH)
    scores = scores[scores["factor"].isin(top_factors)]

    # Pivot to wide: one row per (sample, group), one column per factor.
    wide = scores.pivot_table(index=["sample", "group"], columns="factor",
                               values="value").reset_index()

    print("\nBuilding merged per-sample condition metadata across all 3 layers ...")
    sample_meta = build_merged_sample_metadata()

    wide = wide.merge(sample_meta, on="sample", how="left")
    n_unmatched = wide["Temperature"].isna().sum()
    if n_unmatched:
        print(f"\n[warn] {n_unmatched} sample(s) have no match in any contrast_groups "
              f"file -- these will be dropped from the plots.")
    wide = wide.dropna(subset=["Temperature"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    for factor_x, factor_y in factor_pairs:
        for variable in CONDITION_VARIABLES:
            output_path = OUTPUT_DIR / f"{factor_x}_vs_{factor_y}_{variable}_scatter.png"
            try:
                plot_pair(wide, factor_x, factor_y, variable, color_maps[variable], output_path)
            except OSError as e:
                print(f"  [ERROR] giving up on {output_path.name}: {e}")
                failures.append(output_path.name)

    n_total = len(factor_pairs) * len(CONDITION_VARIABLES)
    n_ok = n_total - len(failures)
    print(f"\nDone. {n_ok}/{n_total} plots written to {OUTPUT_DIR}")
    if failures:
        print(f"[summary] {len(failures)} plot(s) failed to save after retries: {failures}")


if __name__ == "__main__":
    main()