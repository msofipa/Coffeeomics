"""
correlate_single_layer.py
──────────────────────────
Generic "single-layer" Pearson correlation runner: for any omics layer
(metabolomics, proteomics, transcriptomics, ...), correlates that layer's
samples against themselves (symmetric matrix) and saves, per species,
ONLY 3 figures, via build_single_layer_tree_and_heatmaps() in
make_correlations.py:

    1. {species}_pearson_tree                  - dendrogram, WITH the
                                                  Temperature/CO2/Water-Stress
                                                  attribute strips and legend
    2. {species}_pearson_ordered_heatmap       - Pearson r heatmap, samples
                                                  in numeric/alphabetic order.
    3. {species}_gene_overlap_ordered_heatmap  - feature-count-per-pair
                                                  heatmap, same ordering.

"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# MAKE make_correlations.py IMPORTABLE
# ─────────────────────────────────────────────────────────────────────────
# make_correlations.py lives in the integration folder, not next to this
# script, so we add it to sys.path before importing from it.
INTEGRATION_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")
sys.path.append(str(INTEGRATION_DIR))

from integration.make_correlations import (
    safe_name,
    col_to_species,
    set_contrast_groups,                    # loads contrast_groups_<layer>.json per layer
    build_single_layer_tree_and_heatmaps,   # <- added per make_correlations_additions.py
)

# make_correlations.py already does its own sys.path.append(...) to reach
# color_maps.py, so this import resolves through that same entry.
from color_maps import COLOR_MAPS


# ─────────────────────────────────────────────────────────────────────────
# LAYER-SPECIFIC PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────

def flatten_multiindex_columns_to_sample_id(df):
    """
    For files whose columns are a multi-level header
    (Species, Temperature, CO2, Water Stress, Sample ID, ...), keep only
    the LAST level as the column name — that's the short sample ID
    (e.g. '7B', '52C') that col_to_species()/contrast_groups.json expect.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df


# ─────────────────────────────────────────────────────────────────────────
# LAYER DEFINITIONS  ← edit this list to add/remove layers
# ─────────────────────────────────────────────────────────────────────────
# Each entry describes ONE omics layer to correlate against itself.
#
#   name        : short identifier, used only in console messages
#   input_def   : human-readable label used in plot titles / axis labels
#                 (also used to name the output subfolder, via safe_name())
#   path        : path to the data file for this layer
#   loader      : "csv" or "excel"
#   read_kwargs : extra kwargs passed to the pandas reader (index_col etc.)
#   output_dir  : folder where the "<safe_name(input_def)>" subfolder
#                 of figures will be created
#   preprocess  : OPTIONAL callable(df) -> df, applied right after loading.
#                 Use this for anything layer-specific that isn't just a
#                 raw read — e.g. collapsing transcript-level rows to
#                 gene-level (as the notebook did for trans_mapped).
#   show_attribute_strips : OPTIONAL bool, default True. Set False to
#                 draw the dendrogram without the Temperature/CO2/Water-
#                 Stress strips for this layer.
LAYERS = [
    {
        "name": "metabolomics",
        "input_def": "Metabolomics Raw Values",
        "path": r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\data\metabolomics_counts.csv",
        "loader": "csv",
        "read_kwargs": {"index_col": 0, "header": [0, 1, 2, 3, 4]},
        "preprocess": flatten_multiindex_columns_to_sample_id,
        "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\correltion_matrix",
    },

    {
        "name": "proteomics",
        "input_def": "Proteomics Normalized (PyDESeq2, at least 50% and above 10)",
        "path": r"C:\Users\papan\Desktop\Coffeeomics\proteomics\data\prot_normalized_pydeseq2_manual_50_above_10.xlsx",
        "loader": "excel",
        "read_kwargs": {"index_col": 0},
        "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\proteomics\correltion_matrix",
     },
     {
        "name": "transcriptomics",
        "input_def": "Transcriptomics Normalized PyDESeq2",
        "path": r"C:\Users\papan\Desktop\Coffeeomics\transcriptomics\normalizations\trans_pydeseq2_normalized_counts.csv",
        "loader": "csv",
        "read_kwargs": {"index_col": 0},
        "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\transcriptomics\correltion_matrix",
     },
]

# ─────────────────────────────────────────────────────────────────────────
# LOADING / SPECIES-GROUP SETUP
# ─────────────────────────────────────────────────────────────────────────

def load_layer_data(layer):
    """Read a layer's data file according to its declared loader type."""
    path = Path(layer["path"])
    if not path.exists():
        raise FileNotFoundError(f"Data file for layer '{layer['name']}' not found: {path}")

    kwargs = dict(layer.get("read_kwargs", {}))

    if layer["loader"] == "csv":
        df = pd.read_csv(path, **kwargs)
    elif layer["loader"] == "excel":
        df = pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unknown loader '{layer['loader']}' for layer '{layer['name']}'")

    preprocess = layer.get("preprocess")
    if preprocess is not None:
        df = preprocess(df)

    return df


def build_species_groups(contrast_groups):
    """
    Reproduces the species-detection logic from the notebook, built from
    a contrast_groups dict (pass the layer-specific one returned by
    set_contrast_groups(), NOT a stale module-level import).

    Parameters
    ----------
    contrast_groups : dict  contrast group key -> list[str] short sample IDs

    Returns
    -------
    unique_groups        : list[str]  all contrast group keys, in order
    group_to_color       : dict       contrast group key -> RGB color tuple
    all_species          : list[str]  unique species names, sorted
    short_id_to_species  : dict       short sample ID (e.g. "7A") -> species
    """
    unique_groups = list(contrast_groups.keys())

    # Combine two 20-color matplotlib palettes -> 40 distinct colors
    palette_colors = (
        list(plt.colormaps["tab20"].colors) +
        list(plt.colormaps["tab20b"].colors)
    )
    group_to_color = {
        g: palette_colors[i % len(palette_colors)]
        for i, g in enumerate(unique_groups)
    }

    # Group key format: "C. arabica | 25/20°C | 380ppm | without"
    all_species = sorted({g.split(" | ")[0] for g in unique_groups})

    short_id_to_species = {}
    for group, short_ids in contrast_groups.items():
        species = group.split(" | ")[0]
        for sid in short_ids:
            short_id_to_species[sid] = species

    return unique_groups, group_to_color, all_species, short_id_to_species


# ─────────────────────────────────────────────────────────────────────────
# PER-LAYER RUN
# ─────────────────────────────────────────────────────────────────────────

def run_layer(layer):
    """Build and save the within-layer correlation figures for one layer, per species."""
    print(f"\n{'=' * 70}\nLayer: {layer['name']}  ({layer['input_def']})\n{'=' * 70}")

    # Each layer has its own contrast_groups_<layer>.json (layer['name'] must
    # be one of "transcriptomics"/"proteomics"/"metabolomics" — matches the
    # LAYERS entries below). Loading it here (instead of once in main())
    # keeps every layer's species/group detection consistent with its own
    # contrast groups file.
    contrast_groups = set_contrast_groups(layer["name"], base_dir=str(INTEGRATION_DIR))
    unique_groups, group_to_color, all_species, short_id_to_species = build_species_groups(contrast_groups)
    print(f"Detected species for {layer['name']}: {all_species}")

    data = load_layer_data(layer)

    # Map every sample (column) to its species using col_to_species(),
    # which tokenises the column name and matches against short_id_to_species
    sample_list = sorted(data.columns.tolist())
    sample_to_species = {s: col_to_species(s, short_id_to_species) for s in sample_list}

    unmatched_samples = [s for s, sp in sample_to_species.items() if sp is None]
    if unmatched_samples:
        print(f"Warning: {len(unmatched_samples)} sample(s) not assigned a species:")
        for u in unmatched_samples:
            print(f"  {u}")

    # Group samples by species -> {species_name: [sample_col, ...]}
    species_samples = {
        sp: [s for s in sample_list if sample_to_species[s] == sp]
        for sp in all_species
    }
    for sp, slist in species_samples.items():
        print(f"  {sp}: {len(slist)} samples")

    # One output subfolder per layer, named after its INPUT_DEF label
    output_dir = Path(layer["output_dir"])
    layer_folder = output_dir / safe_name(layer["input_def"])
    layer_folder.mkdir(parents=True, exist_ok=True)

    for sp, sp_samples in species_samples.items():
        if len(sp_samples) < 2:
            print(f"Skipping {sp}: fewer than 2 samples.")
            continue

        build_single_layer_tree_and_heatmaps(
            sp=sp,
            sp_samples=sp_samples,
            data=data,
            comparison_folder=layer_folder,
            unique_groups=unique_groups,
            group_to_color=group_to_color,
            COLOR_MAPS=COLOR_MAPS,
            INPUT_DEF=layer["input_def"],
            show_attribute_strips=layer.get("show_attribute_strips", True),
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    for layer in LAYERS:
        run_layer(layer)

    print("\nFinished.")


if __name__ == "__main__":
    main()