"""
Cross-layer Pearson correlation runner for proteomics vs transcriptomics:
correlates FEATURE VALUES between the two layers, per sample pair, via
build_cross_layer_tree_and_heatmaps() in make_correlations.py, and saves,
per species, ONLY 3 figures (same trimmed set as the single-layer runner):

    1. {species}_pearson_tree                  - dendrogram, WITH the
                                                  Temperature/CO2/Water-Stress
                                                  attribute strips and legend
    2. {species}_pearson_ordered_heatmap       - Pearson r heatmap, rows and
                                                  columns ordered by the SAME
                                                  sample-ID order, so the
                                                  diagonal is meaningful
                                                  (sample i vs itself, across
                                                  layers).
    3. {species}_gene_overlap_ordered_heatmap  - feature-count-per-pair
                                                  heatmap, same ordering.

──────────────────────────────────────────────────────────────────────────
IMPORTANT — shared feature space required
──────────────────────────────────────────────────────────────────────────
build_cross_layer_tree_and_heatmaps() computes pearsonr(layer_A[sample1],
layer_B[sample2]) ROW-BY-ROW, so both layers must be indexed on the SAME
feature space. Proteomics and transcriptomics both reduce naturally to
gene-level index (transcripts -> genes via transcript_gene.csv), so this
pair produces real, meaningful overlap.

Metabolomics is handled separately (different feature space — metabolite,
not gene — and no crosswalk is wired in here).
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# MAKE make_correlations.py IMPORTABLE
# ─────────────────────────────────────────────────────────────────────────
INTEGRATION_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")
sys.path.append(str(INTEGRATION_DIR))

from integration.make_correlations import (
    safe_name,
    col_to_species,
    set_contrast_groups_merged,   # merges contrast_groups_<layer>.json across a pair
    build_cross_layer_tree_and_heatmaps,   # <- trimmed cross-layer builder (3 figures)
)

# make_correlations.py already does its own sys.path.append(...) to reach
# color_maps.py, so this import resolves through that same entry.
from color_maps import COLOR_MAPS


# ─────────────────────────────────────────────────────────────────────────
# FEATURE MAPPING (transcript -> gene, to align with proteomics' gene index)
# ─────────────────────────────────────────────────────────────────────────

def map_transcripts_to_genes(trans_df, mapping_path):
    """
    Aggregate transcript-level rows to gene-level (mean per gene), using a
    transcript -> gene crosswalk CSV whose index is the transcript ID and
    which has a 'gene' column.
    """
    corr = pd.read_csv(mapping_path, index_col=0)
    transcript_to_gene = corr["gene"]

    mapped = trans_df.copy()
    mapped.index = mapped.index.map(transcript_to_gene)
    mapped = mapped[mapped.index.notna()]           # drop unmapped transcripts
    mapped = mapped.groupby(mapped.index).mean()     # collapse duplicates
    return mapped


TRANSCRIPT_GENE_MAP = r"C:\Users\papan\Desktop\Coffeeomics\GOA\inputs\transcript_gene.csv"


# ─────────────────────────────────────────────────────────────────────────
# LAYER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────
# Paths / read_kwargs kept IDENTICAL to the verified-working entries in
# single_omics_corr_analysis.py — only feature_mapping is new (needed here,
# not needed for within-layer self-correlation).
PROTEOMICS = {
    "name": "proteomics",
    "input_def": "Proteomics Normalized (PyDESeq2, at least 50% and above 10)",
    "path": r"C:\Users\papan\Desktop\Coffeeomics\proteomics\data\prot_normalized_pydeseq2_manual_50_above_10.xlsx",
    "loader": "excel",
    "read_kwargs": {"index_col": 0},
    "preprocess": None,
    "feature_mapping": None,   # already gene-indexed
}

TRANSCRIPTOMICS = {
    "name": "transcriptomics",
    "input_def": "Transcriptomics Normalized PyDESeq2",
    "path": r"C:\Users\papan\Desktop\Coffeeomics\transcriptomics\normalizations\trans_pydeseq2_normalized_counts.csv",
    "loader": "csv",
    "read_kwargs": {"index_col": 0},
    "preprocess": None,
    "feature_mapping": lambda df: map_transcripts_to_genes(df, TRANSCRIPT_GENE_MAP),
}

PAIR = {
    "first": PROTEOMICS,
    "second": TRANSCRIPTOMICS,
    "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\integration\cross_layer_correltion_matrix",
    "show_attribute_strips": True,
}


# ─────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────

def load_layer(layer_def):
    """Read a layer's data file, apply preprocess, then feature_mapping."""
    path = Path(layer_def["path"])
    if not path.exists():
        raise FileNotFoundError(f"Data file for layer '{layer_def['name']}' not found: {path}")

    kwargs = dict(layer_def.get("read_kwargs", {}))
    if layer_def["loader"] == "csv":
        df = pd.read_csv(path, **kwargs)
    elif layer_def["loader"] == "excel":
        df = pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unknown loader '{layer_def['loader']}' for layer '{layer_def['name']}'")

    preprocess = layer_def.get("preprocess")
    if preprocess is not None:
        df = preprocess(df)

    feature_mapping = layer_def.get("feature_mapping")
    if feature_mapping is not None:
        df = feature_mapping(df)

    return df


def build_species_groups(contrast_groups):
    """
    Reproduces the species-detection logic from the notebook, built from
    a (possibly merged, via set_contrast_groups_merged) contrast_groups dict.
    """
    unique_groups = list(contrast_groups.keys())

    palette_colors = (
        list(plt.colormaps["tab20"].colors) +
        list(plt.colormaps["tab20b"].colors)
    )
    group_to_color = {
        g: palette_colors[i % len(palette_colors)]
        for i, g in enumerate(unique_groups)
    }

    all_species = sorted({g.split(" | ")[0] for g in unique_groups})

    short_id_to_species = {}
    for group, short_ids in contrast_groups.items():
        species = group.split(" | ")[0]
        for sid in short_ids:
            short_id_to_species[sid] = species

    return unique_groups, group_to_color, all_species, short_id_to_species


# ─────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────

def run_pair(pair):
    """Build and save the cross-layer correlation figures for one pair, per species."""
    first_def  = pair["first"]
    second_def = pair["second"]

    print(f"\n{'=' * 70}\n{first_def['input_def']}  vs  {second_def['input_def']}\n{'=' * 70}")

    first_data  = load_layer(first_def)
    second_data = load_layer(second_def)

    # ── Shared feature space (after each layer's optional feature_mapping) ─
    common_features = first_data.index.intersection(second_data.index)
    first_aligned  = first_data.loc[common_features]
    second_aligned = second_data.loc[common_features]

    print(f"  Shared features: {len(common_features)}  "
          f"({len(first_data)} in {first_def['name']}, {len(second_data)} in {second_def['name']})")
    if len(common_features) == 0:
        print("  ⚠  No overlapping features between these two layers. "
              "build_and_save() needs both layers on the same feature index "
              "(e.g. gene) to produce meaningful correlations — check the "
              "transcript_gene.csv crosswalk.")

    # ── Shared samples (columns) ────────────────────────────────────────────
    common_samples = first_aligned.columns.intersection(second_aligned.columns)

    only_in_first  = set(first_aligned.columns) - set(second_aligned.columns)
    only_in_second = set(second_aligned.columns) - set(first_aligned.columns)
    if only_in_first:
        print(f"  In {first_def['name']} only ({len(only_in_first)}): {sorted(only_in_first)}")
    if only_in_second:
        print(f"  In {second_def['name']} only ({len(only_in_second)}): {sorted(only_in_second)}")

    first_aligned  = first_aligned[common_samples]
    second_aligned = second_aligned[common_samples]

    # ── Contrast groups: merge both layers so either naming convention matches ─
    contrast_groups = set_contrast_groups_merged(
        (first_def["name"], second_def["name"]),
        base_dir=str(INTEGRATION_DIR),
    )
    unique_groups, group_to_color, all_species, short_id_to_species = build_species_groups(contrast_groups)
    print(f"  Detected species: {all_species}")

    sample_list = sorted(common_samples.tolist())
    sample_to_species = {s: col_to_species(s, short_id_to_species) for s in sample_list}

    unmatched_samples = [s for s, sp in sample_to_species.items() if sp is None]
    if unmatched_samples:
        print(f"  Warning: {len(unmatched_samples)} sample(s) not assigned a species:")
        for u in unmatched_samples:
            print(f"    {u}")

    species_samples = {
        sp: [s for s in sample_list if sample_to_species[s] == sp]
        for sp in all_species
    }
    for sp, slist in species_samples.items():
        print(f"    {sp}: {len(slist)} samples")

    # One output subfolder for the pair, named after both INPUT_DEF labels
    output_dir = Path(pair["output_dir"])
    comparison_folder = output_dir / (
        f"{safe_name(first_def['input_def'])}__vs__{safe_name(second_def['input_def'])}"
    )
    comparison_folder.mkdir(parents=True, exist_ok=True)

    for sp, sp_samples in species_samples.items():
        if len(sp_samples) < 2:
            print(f"  Skipping {sp}: fewer than 2 samples.")
            continue

        # NOTE: build_cross_layer_tree_and_heatmaps() (a) sorts BOTH the row
        # and column sample order identically for the ordered heatmap (same
        # sp_samples list, same _sort_key, applied to both axes), which is
        # what makes the diagonal meaningful — cell [i, i] is sample i's
        # value in the first layer vs sample i's value in the second layer;
        # and (b) draws the Temperature/CO2/Water-Stress attribute strips
        # beside the dendrogram when show_attribute_strips=True (the
        # default). It saves only 3 figures per species (tree + 2 ordered
        # heatmaps) — the plain/annotated clustermaps are skipped.
        build_cross_layer_tree_and_heatmaps(
            sp=sp,
            sp_samples=sp_samples,
            prot_aligned=first_aligned,
            trans_aligned=second_aligned,
            comparison_folder=comparison_folder,
            unique_groups=unique_groups,
            group_to_color=group_to_color,
            COLOR_MAPS=COLOR_MAPS,
            FIRST_INPUT_DEF=first_def["input_def"],
            SECOND_INPUT_DEF=second_def["input_def"],
            show_attribute_strips=pair.get("show_attribute_strips", True),
            layers=None,  # contrast_groups already merged & set above
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    run_pair(PAIR)
    print("\nFinished.")


if __name__ == "__main__":
    main()