"""
Cross-layer Pearson correlation runner for METABOLOMICS vs proteomics and
METABOLOMICS vs transcriptomics.

Metabolomics has a different native feature space (KEGG compound IDs) from
proteomics/transcriptomics (genes), so unlike the proteomics<->transcriptomics
runner - which aligns on gene ID directly - this script aligns all three
layers on a COMMON, COARSER feature space: KEGG PATHWAY ID (the 5-digit
numeric suffix shared across ko/map/organism-specific pathway prefixes,
e.g. "00010" for ko00010 / map00010 / cce00010 - Glycolysis).

Bridge used for each layer:
    - Metabolomics : metabolite name -> KEGG compound ID (via
                      metabolites_kegg_ids.csv) -> KEGG pathway ID(s)
                      (via the KEGG REST /link/pathway/cpd: endpoint).
    - Proteomics    : gene -> KO (via GhostKOALA output, user_ko.txt,
                      joined through transcript_gene.csv) -> KEGG pathway
                      ID(s) (via the KEGG REST /link/pathway/ko: endpoint).
    - Transcriptomics: transcript -> gene (transcript_gene.csv) -> KO ->
                      pathway, using the same GhostKOALA output as above.

A feature (gene or compound) that maps to multiple pathways contributes a
row to EACH of those pathways; pathway-level values are then the mean of
all features assigned to that pathway.

Saves, per species, per pair, the same trimmed 3-figure set as the other
cross-layer runner:
    1. {species}_pearson_tree
    2. {species}_pearson_ordered_heatmap
    3. {species}_gene_overlap_ordered_heatmap   (here: pathway-overlap count)

──────────────────────────────────────────────────────────────────────────
BEFORE RUNNING - fill in / verify the path marked TODO below:
──────────────────────────────────────────────────────────────────────────
1. METABOLOMICS_DATA_PATH - your metabolomics abundance matrix, same
   layout convention as the proteomics/transcriptomics files (features as
   rows, samples as columns).

USER_KO_PATH and TRANSCRIPT_GENE_MAP are already set to your GhostKOALA
output and transcript-gene crosswalk - double check those paths match
where you actually saved user_ko.txt.

Also requires METABOLITES_KEGG_IDS_PATH (metabolites_kegg_ids.csv) to
already exist, and rows with an empty kegg_id (still flagged "needs
manual review") are excluded automatically - fill those in first if you
want them included.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import requests

# ─────────────────────────────────────────────────────────────────────────
# MAKE make_correlations.py IMPORTABLE
# ─────────────────────────────────────────────────────────────────────────
INTEGRATION_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")
sys.path.append(str(INTEGRATION_DIR))

from integration.make_correlations import (
    safe_name,
    col_to_species,
    set_contrast_groups_merged,
    build_cross_layer_tree_and_heatmaps,
)
from color_maps import COLOR_MAPS


# ─────────────────────────────────────────────────────────────────────────
# CONFIG - paths
# ─────────────────────────────────────────────────────────────────────────
TRANSCRIPT_GENE_MAP = r"C:\Users\papan\Desktop\Coffeeomics\GOA\inputs\transcript_gene.csv"
METABOLITES_KEGG_IDS_PATH = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\metabolites_kegg_ids.csv"

# GhostKOALA output: two columns, transcript_id <TAB> KO (blank when no hit).
# Keyed by transcript (GSCOC_T...), so it's joined through TRANSCRIPT_GENE_MAP
# to reach gene level, then each KO is resolved to its KEGG pathway(s) via
# the REST API (GhostKOALA gives KO only, not pathway, directly).
USER_KO_PATH = r"C:\Users\papan\Desktop\Coffeeomics\GOA\inputs\GhostKOALA.txt"

KEGG_LINK_PATHWAY_FROM_KO_URL = "https://rest.kegg.jp/link/pathway/ko:{}"

# TODO: point this at your metabolomics abundance matrix (features x samples).
METABOLOMICS_DATA_PATH = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\data\metabolomics_counts.csv"

KEGG_LINK_PATHWAY_URL = "https://rest.kegg.jp/link/pathway/cpd:{}"
REQUEST_DELAY = 0.35  # be polite to the free KEGG API


# ─────────────────────────────────────────────────────────────────────────
# PATHWAY ID NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────

def pathway_suffix(pathway_id):
    """
    Reduce a pathway identifier to its 5-digit numeric suffix, which is
    shared across the generic "map#####", "ko#####", and any
    organism-specific ("cce#####" etc.) forms of the same pathway. This is
    what lets gene-derived (ko/map) and compound-derived (map) pathway IDs
    be compared directly.
    """
    digits = "".join(ch for ch in pathway_id if ch.isdigit())
    return digits[-5:] if len(digits) >= 5 else digits


# ─────────────────────────────────────────────────────────────────────────
# GENE -> PATHWAY (for proteomics, and transcriptomics after transcript->gene)
# ─────────────────────────────────────────────────────────────────────────

def load_gene_to_ko(user_ko_path, transcript_gene_path):
    """
    Parse GhostKOALA's output (transcript_id <TAB> KO, blank when no hit) and
    join through transcript_gene.csv to build {gene_id: {KO, ...}}. A gene
    can carry multiple transcripts, and each may resolve to a different (or
    no) KO, so KOs are unioned per gene.
    """
    transcript_to_gene = pd.read_csv(transcript_gene_path, index_col=0)["gene"]

    gene_to_kos = defaultdict(set)
    with open(user_ko_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            transcript_id = parts[0].strip()
            ko = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            if not ko:
                continue
            gene_id = transcript_to_gene.get(transcript_id)
            if gene_id is not None and pd.notna(gene_id):
                gene_to_kos[gene_id].add(ko)

    return dict(gene_to_kos)


def fetch_ko_pathways(ko, cache):
    """Query KEGG's /link/pathway/ko:<id> endpoint, with a simple in-memory cache."""
    if ko in cache:
        return cache[ko]

    resp = requests.get(KEGG_LINK_PATHWAY_FROM_KO_URL.format(ko), timeout=15)
    resp.raise_for_status()

    suffixes = set()
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        _, pathway = line.split("\t")
        suffixes.add(pathway_suffix(pathway.replace("path:", "")))
    suffixes.discard("")

    cache[ko] = suffixes
    time.sleep(REQUEST_DELAY)
    return suffixes


def build_gene_to_pathway_map(gene_to_kos):
    """{gene_id: {pathway_suffix, ...}}, resolved from each gene's KO set via KEGG."""
    cache = {}
    gene_to_pathways = {}
    unique_kos = {ko for kos in gene_to_kos.values() for ko in kos}
    print(f"  Resolving {len(unique_kos)} unique KOs to pathways via KEGG API...")

    for gene, kos in gene_to_kos.items():
        pathways = set()
        for ko in kos:
            try:
                pathways |= fetch_ko_pathways(ko, cache)
            except requests.RequestException as e:
                print(f"  Warning: pathway lookup failed for {ko}: {e}")
        if pathways:
            gene_to_pathways[gene] = pathways

    return gene_to_pathways


def map_transcripts_to_genes(trans_df, mapping_path):
    """Aggregate transcript-level rows to gene-level (mean per gene)."""
    corr = pd.read_csv(mapping_path, index_col=0)
    transcript_to_gene = corr["gene"]

    mapped = trans_df.copy()
    mapped.index = mapped.index.map(transcript_to_gene)
    mapped = mapped[mapped.index.notna()]
    mapped = mapped.groupby(mapped.index).mean()
    return mapped


# ─────────────────────────────────────────────────────────────────────────
# COMPOUND -> PATHWAY (for metabolomics)
# ─────────────────────────────────────────────────────────────────────────

def load_metabolite_to_kegg_id(kegg_ids_path):
    """
    Read metabolites_kegg_ids.csv and return {metabolite_name: kegg_id},
    excluding rows that still need manual review (empty kegg_id).
    """
    df = pd.read_csv(kegg_ids_path)
    resolved = df[df["kegg_id"].notna() & (df["kegg_id"] != "")]
    skipped = df[df["kegg_id"].isna() | (df["kegg_id"] == "")]

    if len(skipped):
        print(f"  Skipping {len(skipped)} unresolved metabolite(s) (no kegg_id): "
              f"{skipped['query'].tolist()}")

    return dict(zip(resolved["query"], resolved["kegg_id"]))


def fetch_compound_pathways(kegg_id, cache):
    """Query KEGG's /link/pathway/cpd:<id> endpoint, with a simple in-memory cache."""
    if kegg_id in cache:
        return cache[kegg_id]

    url = KEGG_LINK_PATHWAY_URL.format(kegg_id)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    suffixes = set()
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        _, pathway = line.split("\t")
        suffixes.add(pathway_suffix(pathway.replace("path:", "")))
    suffixes.discard("")

    cache[kegg_id] = suffixes
    time.sleep(REQUEST_DELAY)
    return suffixes


def load_compound_to_pathway_map(name_to_kegg_id):
    """Build {kegg_id: {pathway_suffix, ...}} for every resolved compound."""
    cache = {}
    kegg_id_to_pathways = {}
    for kegg_id in set(name_to_kegg_id.values()):
        try:
            kegg_id_to_pathways[kegg_id] = fetch_compound_pathways(kegg_id, cache)
        except requests.RequestException as e:
            print(f"  Warning: pathway lookup failed for {kegg_id}: {e}")
            kegg_id_to_pathways[kegg_id] = set()
    return kegg_id_to_pathways


# ─────────────────────────────────────────────────────────────────────────
# SHARED PATHWAY-LEVEL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────

def aggregate_by_pathway(df, feature_to_pathways):
    """
    Explode each feature (row) across every pathway it belongs to, then
    collapse to one row per pathway (mean across all features assigned to
    it). A feature with no pathway assignment is dropped.
    """
    exploded_index = []
    exploded_rows = []
    for feature, row in df.iterrows():
        for pw in feature_to_pathways.get(feature, ()):
            exploded_index.append(pw)
            exploded_rows.append(row)

    if not exploded_rows:
        return pd.DataFrame(columns=df.columns)

    long_df = pd.DataFrame(exploded_rows, index=exploded_index, columns=df.columns)
    return long_df.groupby(long_df.index).mean()


def metabolomics_feature_mapping(df, name_to_kegg_id, kegg_id_to_pathways):
    """Metabolite name -> KEGG compound ID -> pathway, aggregated by pathway."""
    renamed = df.copy()
    renamed.index = renamed.index.map(name_to_kegg_id)
    renamed = renamed[renamed.index.notna()]
    return aggregate_by_pathway(renamed, kegg_id_to_pathways)


def transcriptomics_feature_mapping(df, gene_to_pathways):
    gene_df = map_transcripts_to_genes(df, TRANSCRIPT_GENE_MAP)
    return aggregate_by_pathway(gene_df, gene_to_pathways)


# ─────────────────────────────────────────────────────────────────────────
# LOADING (generic, same shape as the proteomics/transcriptomics runner)
# ─────────────────────────────────────────────────────────────────────────

def load_layer(layer_def):
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
# RUN (one pair)
# ─────────────────────────────────────────────────────────────────────────

def run_pair(pair):
    first_def = pair["first"]
    second_def = pair["second"]

    print(f"\n{'=' * 70}\n{first_def['input_def']}  vs  {second_def['input_def']}\n{'=' * 70}")

    first_data = load_layer(first_def)
    second_data = load_layer(second_def)

    common_features = first_data.index.intersection(second_data.index)
    first_aligned = first_data.loc[common_features]
    second_aligned = second_data.loc[common_features]

    print(f"  Shared KEGG pathways: {len(common_features)}  "
          f"({len(first_data)} in {first_def['name']}, {len(second_data)} in {second_def['name']})")
    if len(common_features) == 0:
        print("  ⚠  No overlapping pathways between these two layers - check the "
              "gene/compound -> pathway crosswalks above.")

    common_samples = first_aligned.columns.intersection(second_aligned.columns)

    only_in_first = set(first_aligned.columns) - set(second_aligned.columns)
    only_in_second = set(second_aligned.columns) - set(first_aligned.columns)
    if only_in_first:
        print(f"  In {first_def['name']} only ({len(only_in_first)}): {sorted(only_in_first)}")
    if only_in_second:
        print(f"  In {second_def['name']} only ({len(only_in_second)}): {sorted(only_in_second)}")

    first_aligned = first_aligned[common_samples]
    second_aligned = second_aligned[common_samples]

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

    output_dir = Path(pair["output_dir"])
    comparison_folder = output_dir / (
        f"{safe_name(first_def['input_def'])}__vs__{safe_name(second_def['input_def'])}"
    )
    comparison_folder.mkdir(parents=True, exist_ok=True)

    for sp, sp_samples in species_samples.items():
        if len(sp_samples) < 2:
            print(f"  Skipping {sp}: fewer than 2 samples.")
            continue

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
            layers=None,
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("Building gene -> KO crosswalk from GhostKOALA output...")
    gene_to_kos = load_gene_to_ko(USER_KO_PATH, TRANSCRIPT_GENE_MAP)
    print(f"  {len(gene_to_kos)} genes with at least one KO assignment.")

    print("Resolving gene -> pathway via KO...")
    gene_to_pathways = build_gene_to_pathway_map(gene_to_kos)
    print(f"  {len(gene_to_pathways)} genes with at least one KEGG pathway assignment.")

    print("Building metabolite -> compound -> pathway crosswalk...")
    name_to_kegg_id = load_metabolite_to_kegg_id(METABOLITES_KEGG_IDS_PATH)
    kegg_id_to_pathways = load_compound_to_pathway_map(name_to_kegg_id)
    print(f"  {len(name_to_kegg_id)} metabolites resolved to compound IDs; "
          f"{sum(1 for v in kegg_id_to_pathways.values() if v)} of those have a pathway assignment.")

    PROTEOMICS_PW = {
        "name": "proteomics",
        "input_def": "Proteomics Normalized (PyDESeq2, at least 50% and above 10)",
        "path": r"C:\Users\papan\Desktop\Coffeeomics\proteomics\data\prot_normalized_pydeseq2_manual_50_above_10.xlsx",
        "loader": "excel",
        "read_kwargs": {"index_col": 0},
        "preprocess": None,
        "feature_mapping": lambda df: aggregate_by_pathway(df, gene_to_pathways),
    }

    TRANSCRIPTOMICS_PW = {
        "name": "transcriptomics",
        "input_def": "Transcriptomics Normalized PyDESeq2",
        "path": r"C:\Users\papan\Desktop\Coffeeomics\transcriptomics\normalizations\trans_pydeseq2_normalized_counts.csv",
        "loader": "csv",
        "read_kwargs": {"index_col": 0},
        "preprocess": None,
        "feature_mapping": lambda df: transcriptomics_feature_mapping(df, gene_to_pathways),
    }

    METABOLOMICS = {
        "name": "metabolomics",
        "input_def": "Metabolomics Normalized",
        "path": METABOLOMICS_DATA_PATH,
        "loader": "excel",  # change to "csv" if your file is .csv
        "read_kwargs": {"index_col": 0},
        "preprocess": None,
        "feature_mapping": lambda df: metabolomics_feature_mapping(df, name_to_kegg_id, kegg_id_to_pathways),
    }

    pairs = [
        {
            "first": METABOLOMICS,
            "second": PROTEOMICS_PW,
            "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\integration\cross_layer_correltion_matrix",
            "show_attribute_strips": True,
        },
        {
            "first": METABOLOMICS,
            "second": TRANSCRIPTOMICS_PW,
            "output_dir": r"C:\Users\papan\Desktop\Coffeeomics\integration\cross_layer_correltion_matrix",
            "show_attribute_strips": True,
        },
    ]

    for pair in pairs:
        run_pair(pair)

    print("\nFinished.")


if __name__ == "__main__":
    main()