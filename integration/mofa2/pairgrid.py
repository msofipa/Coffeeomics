"""
plot_factor_pairgrid.py
------------------------------
Consolidates all pairwise comparisons among the top 5 MOFA factors into a
single seaborn PairGrid per (species, condition variable) -- 2 species x 3
variables = 6 images total, instead of the 30 separate scatter files from
plot_factor_scatter_pairs.py. Each grid shows all 10 factor pairs (lower
triangle only, corner=True) plus per-factor histograms on the diagonal,
colored by the same condition variable throughout.

Self-contained for the same reason as the other plotting scripts (not
coupled to whatever filename the beeswarm/scatter scripts end up saved
under locally) -- the small amount of shared logic is duplicated here.

Inputs:
  - variance_explained.csv   (output of fit_mofa_model.py)
  - factor_scores_long.csv   (output of extract_factor_scores.py)
  - contrast_groups_{transcriptomics,proteomics,metabolomics}.json
  - color_maps.py            (imported directly from BASE_DIR)

>>> Dependency caveat <<<
Same axis-swap-bug caveat as the other two scripts: if
variance_explained.csv still has only ~3 distinct "factor" values, the
top-5 ranking here is unreliable. Use MANUAL_TOP_FACTORS to bypass.

>>> Design choice <<<
One grid per species (not combined into one grid with species and
variable both encoded) -- this preserves the species-panel convention
used in the beeswarm/scatter scripts. If you'd rather overlay both
species in a single grid per variable using marker shape for species,
that's a different (and reasonable) design; ask and this can be adapted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

OUTPUT_DIR = BASE_DIR / "integration" / "mofa2" / "factor_pairgrid_plots"

N_TOP_FACTORS = 5
MANUAL_TOP_FACTORS: list[str] | None = None

CONDITION_VARIABLES = ["Temperature", "CO2_ppm", "Water_Stress"]

DOT_SIZE = 30
PANEL_HEIGHT = 2.2  # inches per grid cell; total figure grows with n_factors

# =========================================================================
# Shared loaders (duplicated across the three plotting scripts by design)
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
              f"bug. This ranking is almost certainly wrong. Set "
              f"MANUAL_TOP_FACTORS above to bypass.")
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
        f"scanning). Try pausing sync/scanning for this folder, or set "
        f"OUTPUT_DIR outside OneDrive-synced Desktop, then rerun."
    ) from last_err


# =========================================================================
# Plotting
# =========================================================================


def plot_pairgrid(df_species: pd.DataFrame, factors: list[str], variable: str,
                   color_map: dict, species: str, output_path: Path) -> None:
    missing_from_map = sorted(set(df_species[variable].dropna().unique()) - set(color_map.keys()))
    extended_palette = dict(color_map)
    if missing_from_map:
        print(f"  [warn] {species}/{variable}: value(s) not in COLOR_MAPS, "
              f"plotted grey: {missing_from_map}")
        for level in missing_from_map:
            extended_palette.setdefault(level, "#999999")
    hue_order = list(extended_palette.keys())

    g = sns.PairGrid(df_species, vars=factors, hue=variable, hue_order=hue_order,
                      palette=extended_palette, corner=True, height=PANEL_HEIGHT)
    g.map_offdiag(sns.scatterplot, s=DOT_SIZE, edgecolor="black", linewidth=0.4)
    g.map_diag(sns.histplot, multiple="stack", edgecolor="black", linewidth=0.3)

    for i in range(len(factors)):
        for j in range(len(factors)):
            ax = g.axes[i, j]
            if ax is None:
                continue
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
            if i != j:
                ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)

    g.add_legend(title=variable)
    g.fig.suptitle(f"{species}: pairwise factor scores colored by {variable}", y=1.02)
    savefig_with_retry(g.fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
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

    wide = scores.pivot_table(index=["sample", "group"], columns="factor",
                               values="value").reset_index()
    # Preserve the requested factor order (pivot_table sorts columns alphabetically,
    # which breaks e.g. Factor2/Factor10 ordering).
    wide = wide[["sample", "group"] + top_factors]

    print("\nBuilding merged per-sample condition metadata across all 3 layers ...")
    sample_meta = build_merged_sample_metadata()

    wide = wide.merge(sample_meta, on="sample", how="left")
    n_unmatched = wide["Temperature"].isna().sum()
    if n_unmatched:
        print(f"\n[warn] {n_unmatched} sample(s) have no match in any contrast_groups "
              f"file -- these will be dropped from the plots.")
    wide = wide.dropna(subset=["Temperature"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    species_list = sorted(wide["group"].unique())
    failures = []
    for species in species_list:
        df_species = wide[wide["group"] == species]
        for variable in CONDITION_VARIABLES:
            safe_species = species.replace(" ", "_").replace(".", "")
            output_path = OUTPUT_DIR / f"pairgrid_{safe_species}_{variable}.png"
            try:
                plot_pairgrid(df_species, top_factors, variable, color_maps[variable],
                               species, output_path)
            except OSError as e:
                print(f"  [ERROR] giving up on {output_path.name}: {e}")
                failures.append(output_path.name)

    n_total = len(species_list) * len(CONDITION_VARIABLES)
    n_ok = n_total - len(failures)
    print(f"\nDone. {n_ok}/{n_total} plots written to {OUTPUT_DIR}")
    if failures:
        print(f"[summary] {len(failures)} plot(s) failed to save after retries: {failures}")


if __name__ == "__main__":
    main()