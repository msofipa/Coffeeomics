"""
plot_top_factors_beeswarm.py
------------------------------
For the top 5 MOFA factors (by summed variance explained across
views/groups), generates beeswarm plots of factor scores faceted by
species (group), for each of three condition variables: Temperature,
CO2_ppm, Water_Stress. Colors come from the project's own
color_maps.py -- imported directly, not duplicated here, so this script
never drifts out of sync with it.

Inputs:
  - variance_explained.csv   (output of fit_mofa_model.py)
  - factor_scores_long.csv   (output of extract_factor_scores.py)
  - contrast_groups_{transcriptomics,proteomics,metabolomics}.json
  - color_maps.py            (imported directly from BASE_DIR)

>>> Dependency caveat <<<
Top-5 ranking is only as good as variance_explained.csv. If the
axis-swap bug (3 apparent "factors" matching view count instead of 15
actual factors) is still present in that file, the ranking here will be
wrong. Re-run summarize_variance_explained() with the fix first, or set
MANUAL_TOP_FACTORS below to bypass the ranking entirely.

>>> Swarm layout <<<
Point placement uses seaborn.swarmplot (true non-overlap swarm
algorithm), not a hand-rolled jitter -- this is what should make the
packing/shape match the Shiny app's plotFactorBeeswarm output. If
seaborn can't fit all points at the given marker size, it prints its own
warning and lets some points overlap; if you see that warning, lower
DOT_SIZE.

>>> Cross-layer metadata caveat <<<
A sample ID's condition (temperature/CO2/water_stress) is looked up by
inverting all three contrast_groups_*.json files and merging them. Per
the project's own documented caveat, a sample ID from one layer has no
guaranteed relationship to the same-looking ID in another layer --  so
if the same ID maps to DIFFERENT conditions across layers, that is
flagged explicitly as a conflict (printed, not silently resolved) rather
than picking one arbitrarily and hiding the discrepancy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def savefig_with_retry(fig, path: Path, attempts: int = 4, delay: float = 1.5, **kwargs) -> None:
    """
    Retries fig.savefig() on OSError. On Windows, a working save that fails
    a few calls later at the identical path pattern (as opposed to failing
    immediately/consistently) is almost always a transient file lock --
    OneDrive sync, Windows Defender real-time scanning, or a preview
    handler -- rather than a code bug, especially for paths under
    Desktop\\Coffeeomics, which OneDrive backs up by default. Retrying with
    a short delay resolves that; a genuine code/path bug will fail the
    same way every attempt.
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
        f"scanning, or a preview handler holding the file open) given the "
        f"identical code path succeeded for earlier factors. Try pausing "
        f"OneDrive sync / Defender real-time protection for this folder, "
        f"or set OUTPUT_DIR to a path outside OneDrive-synced Desktop, "
        f"then rerun."
    ) from last_err

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

OUTPUT_DIR = BASE_DIR / "integration" / "mofa2" / "factor_beeswarm_plots"

N_TOP_FACTORS = 5

# Set to e.g. ["Factor3", "Factor7", ...] to bypass the variance_explained.csv
# ranking entirely (useful while the axis-swap bug is unresolved).
MANUAL_TOP_FACTORS: list[str] | None = None

# Must match the keys inside COLOR_MAPS in color_maps.py
CONDITION_VARIABLES = ["Temperature", "CO2_ppm", "Water_Stress"]

DOT_SIZE = 4
FIGSIZE = (10, 6)

# =========================================================================
# Load color_maps.py directly (no duplication of its contents here)
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


# =========================================================================
# Top-factor selection
# =========================================================================


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
              f"distinct 'factor' values -- this matches the count of omics views "
              f"(3), not the ~15 real MOFA factors the model was trained with. "
              f"This is consistent with the known axis-swap bug in "
              f"summarize_variance_explained() (view axis mislabeled as factor "
              f"axis), meaning the ranking above is almost certainly wrong -- "
              f"'Factor1/2/3' here may actually correspond to views, not real "
              f"factors. Fix that reshape and rerun fit_mofa_model.py's "
              f"summarize_variance_explained(), or set MANUAL_TOP_FACTORS above "
              f"to bypass this ranking entirely with factor names you trust "
              f"(e.g. from inspect_mofa_output.py).")
    elif len(top) < n:
        print(f"\n[warn] only {len(top)} distinct factor(s) found in "
              f"{variance_csv.name}, fewer than the requested {n}.")
    return top


# =========================================================================
# Per-sample condition metadata: invert + merge all three contrast_groups files
# =========================================================================


def invert_contrast_groups(path: Path) -> pd.DataFrame:
    """
    condition string "species | temperature | co2 | water_stress" ->
    [sample_id, ...]  inverted to a per-sample table.
    Same parsing convention as load_sample_metadata() in build_mofa_input.py.
    """
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
    """
    Merge per-sample condition metadata across all three layer-specific
    contrast_groups files. If the same sample ID maps to different
    conditions across layers, that conflict is printed explicitly rather
    than silently resolved.
    """
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
            # No silent resolution -- flagged below, first layer's values used
            # only so the rest of the pipeline can still run.
        chosen = next(iter(values_by_layer.values()))
        record = {"sample": sample_id, **chosen}
        merged_records.append(record)

    if conflicts:
        print(f"\n[CONFLICT] {len(conflicts)} sample ID(s) have DIFFERENT conditions "
              f"across layers -- values used are from the first layer found; "
              f"verify these manually before trusting their plotted color:")
        for sample_id, values_by_layer in conflicts:
            print(f"  {sample_id}: {values_by_layer}")

    return pd.DataFrame(merged_records)


# =========================================================================
# Plotting
# =========================================================================


def plot_one(factor_df: pd.DataFrame, factor_name: str, variable: str,
             color_map: dict, output_path: Path) -> None:
    """
    Uses seaborn.swarmplot for actual non-overlap point placement (same
    algorithm family as R's beeswarm package, which MOFA2's
    plotFactorBeeswarm calls) instead of a hand-rolled jitter -- this is
    what makes the packing/tapering match the Shiny app's output rather
    than looking like a crude two-column zigzag.
    """
    species_list = sorted(factor_df["group"].unique())
    fig, axes = plt.subplots(1, len(species_list), figsize=FIGSIZE, sharey=True)
    if len(species_list) == 1:
        axes = [axes]

    hue_order = list(color_map.keys())
    extended_palette = dict(color_map)

    for ax, species in zip(axes, species_list):
        sub = factor_df[factor_df["group"] == species].copy()

        missing_from_map = sorted(set(sub[variable].dropna().unique()) - set(color_map.keys()))
        if missing_from_map:
            print(f"  [warn] {factor_name}/{variable}/{species}: value(s) not in "
                  f"COLOR_MAPS, plotted grey: {missing_from_map}")
            for level in missing_from_map:
                extended_palette.setdefault(level, "#999999")
                if level not in hue_order:
                    hue_order.append(level)

        sub["_x"] = species
        sns.swarmplot(
            data=sub, x="_x", y="value", hue=variable,
            hue_order=hue_order, palette=extended_palette,
            size=DOT_SIZE, edgecolor="black", linewidth=0.4,
            ax=ax, legend=False,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
        ax.set_xlabel("")
        ax.set_xticklabels([species])

    axes[0].set_ylabel(factor_name)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=level,
                          markerfacecolor=color, markeredgecolor="black", markersize=8)
               for level, color in extended_palette.items()]
    fig.legend(handles=handles, title=variable, loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{factor_name} colored by {variable}")
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

    scores = pd.read_csv(FACTOR_SCORES_LONG_PATH)
    scores = scores[scores["factor"].isin(top_factors)]

    print("\nBuilding merged per-sample condition metadata across all 3 layers ...")
    sample_meta = build_merged_sample_metadata()

    merged = scores.merge(sample_meta, on="sample", how="left")
    n_unmatched = merged["Temperature"].isna().sum()
    if n_unmatched:
        n_samples_unmatched = merged.loc[merged["Temperature"].isna(), "sample"].nunique()
        print(f"\n[warn] {n_samples_unmatched} sample(s) ({n_unmatched} rows) in "
              f"factor_scores_long.csv have no match in any contrast_groups file "
              f"-- these will be dropped from the plots.")
    merged = merged.dropna(subset=["Temperature"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    for factor_name in top_factors:
        factor_df = merged[merged["factor"] == factor_name]
        for variable in CONDITION_VARIABLES:
            output_path = OUTPUT_DIR / f"{factor_name}_{variable}_beeswarm.png"
            try:
                plot_one(factor_df, factor_name, variable, color_maps[variable], output_path)
            except OSError as e:
                # Isolated per-plot so a persistent file lock on one output
                # doesn't abort plots that would otherwise succeed.
                print(f"  [ERROR] giving up on {output_path.name}: {e}")
                failures.append(output_path.name)

    n_total = len(top_factors) * len(CONDITION_VARIABLES)
    n_ok = n_total - len(failures)
    print(f"\nDone. {n_ok}/{n_total} plots written to {OUTPUT_DIR}")
    if failures:
        print(f"[summary] {len(failures)} plot(s) failed to save after retries: {failures}")


if __name__ == "__main__":
    main()