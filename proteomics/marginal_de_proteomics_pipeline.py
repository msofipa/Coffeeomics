"""
Marginal (pooled) Differential Expression pipeline — Coffeeomics proteomics.

Companion to de_proteomics_pipeline.py's fully-factorial per-comparison runs
(one small_df per pre-split pairwise contrast, factor auto-detected). Here,
for a given factor of interest (temperature, co2, or condition), samples are
pooled across all levels of the two remaining (non-focal) factors, WITHIN
each species — yielding one aggregated sample group per (species, factor
level). Pairwise DE contrasts are then run between these pooled groups, same
species, same factor, using the same PyDESeq2 workflow and significance
criteria as de_proteomics_pipeline.py.

*** DATA SOURCES ***
Sample metadata (species/temperature/co2/condition) comes directly from
contrast_groups_proteomics.json, a flat dict of
"Species | Temperature | CO2_ppm | Water_Stress" → [sample_id, ...] — the
same label format as transcriptomics' contrast_groups.json. This replaces
the earlier, less reliable approach of inferring metadata from each
small_df's column MultiIndex.

Protein abundance values still come from INPUTS_FOLDER's small_dfs (there
is no single master abundance matrix for proteomics, unlike transcriptomics'
INPUT_DATA). Every .csv in INPUTS_FOLDER is read (there's no whitelist file
anymore — contrast_groups_proteomics.json replaced valid_contrasts.csv) and
merged into one master abundance matrix, keyed by the sample IDs in each
file's column MultiIndex (level 0). See build_master_abundance.

Recovery ("Rec") is treated as a level of "temperature" (post-stress
recovery timepoint), matching the convention confirmed for transcriptomics.
Recovery-condition samples are excluded prior to pooling specifically for
the "temperature-collapsed" analyses — i.e. whenever temperature is one of
the two non-focal factors being pooled away (factor of interest = co2 or
condition). When temperature itself is the factor of interest, "Rec" is a
real level being contrasted and is never excluded. Verify RECOVERY_LEVEL
against your actual data.

*** ASSUMPTION FLAGGED FOR REVIEW ***
"condition" is assumed to be the proteomics equivalent of transcriptomics'
Water_Stress factor. Confirm this matches your experimental design.
"""

import os
import json
import logging
from itertools import combinations

import pandas as pd
from plyer import notification
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\papan\Desktop\Coffeeomics\proteomics"

INPUTS_FOLDER = os.path.join(
    BASE_DIR,
    r"small_dfs_normalized\small_dfs_filtered_normalized_above_10"
)

CONTRAST_JSON = os.path.join(BASE_DIR, "contrast_groups_proteomics.json")

# All marginal-analysis output (per-factor groupings JSON + per-factor DE
# results Excel files) is written into this one folder, one pair of files
# per factor of interest.
MARGINAL_DIR = os.path.join(BASE_DIR, "de_raw_results", "marginal")

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
P_VALUE_THRESHOLD = 0.01
LFC_THRESHOLD = 1
MIN_PROTEIN_SUM = 10       # same low-abundance prefilter as de_proteomics_pipeline.py
REMOVED_SAMPLES = []

# ─── METADATA STRUCTURE ───────────────────────────────────────────────────────
# Order of attributes encoded in each contrast_groups_proteomics.json label,
# pipe-separated: "Species | Temperature | CO2_ppm | Water_Stress"
LABEL_ATTR_COLS = ["species", "temperature", "co2", "condition"]
LABEL_SEP = " | "
SPECIES_COL = "species"

# Factors eligible to be the "factor of interest" for a marginal analysis.
# species is never a factor of interest — pooling always happens within species.
MARGINAL_FACTORS = ["temperature", "co2", "condition"]

# Exact label used for the post-stress recovery level within temperature.
# *** VERIFY against your actual data — adjust to match exactly. ***
RECOVERY_LEVEL = "Rec"
TEMPERATURE_COL = "temperature"

# Factors of interest for which recovery-condition samples are excluded prior
# to pooling — i.e. whenever temperature (which contains the "Rec" level) is
# one of the two non-focal factors being pooled/collapsed away. This is the
# "temperature-collapsed analysis": factor of interest ≠ temperature.
EXCLUDE_RECOVERY_FOR_FACTORS = {"co2", "condition"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def parse_label(label: str) -> dict[str, str]:
    """'C. arabica | 25/20°C | 380ppm | without' → {'species': ..., ...}"""
    parts = [p.strip() for p in label.split(LABEL_SEP)]
    if len(parts) != len(LABEL_ATTR_COLS):
        raise ValueError(
            f"Label has {len(parts)} fields, expected {len(LABEL_ATTR_COLS)}: {label!r}"
        )
    return dict(zip(LABEL_ATTR_COLS, parts))


def load_sample_metadata() -> pd.DataFrame:
    """Build a sample-metadata table straight from contrast_groups_proteomics.json."""
    with open(CONTRAST_JSON) as f:
        contrast_groups: dict[str, list[str]] = json.load(f)

    sample_meta: dict[str, dict[str, str]] = {}
    conflicts = 0

    for label, sample_ids in contrast_groups.items():
        attrs = parse_label(label)
        for sample_id in sample_ids:
            if sample_id not in sample_meta:
                sample_meta[sample_id] = attrs
            elif sample_meta[sample_id] != attrs:
                conflicts += 1
                log.warning(
                    "Sample %s appears under conflicting labels — keeping first-seen: %s vs %s",
                    sample_id, sample_meta[sample_id], attrs,
                )

    if conflicts:
        log.warning("%d sample(s) had conflicting metadata across labels — review contrast_groups_proteomics.json.", conflicts)

    meta_df = pd.DataFrame.from_dict(sample_meta, orient="index")
    meta_df.index.name = "sample"
    log.info("Loaded metadata for %d samples from %s", len(meta_df), CONTRAST_JSON)
    return meta_df


# ─── STEP 1: BUILD MASTER ABUNDANCE MATRIX ─────────────────────────────────────

def build_master_abundance() -> pd.DataFrame:
    """
    Read every .csv in INPUTS_FOLDER and merge into one master protein
    abundance matrix (proteins × all unique samples seen).

    Sample IDs are taken from level 0 of each file's column MultiIndex —
    the other MultiIndex levels (species/temperature/co2/condition) are
    ignored here, since metadata now comes from contrast_groups_proteomics.json.

    Assumption: different files may cover different protein subsets for the
    same sample; the master matrix is the union (outer join) of all proteins
    seen, and any sample/protein combination never observed is filled with 0
    prior to rounding to int — consistent with how de_proteomics_pipeline.py
    already treats missing/absent measurements.
    """
    sample_series: dict[str, pd.Series] = {}

    for file in os.listdir(INPUTS_FOLDER):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(INPUTS_FOLDER, file)
        small_df = pd.read_csv(file_path, index_col=0, header=[0, 1, 2, 3, 4])

        sample_ids = small_df.columns.get_level_values(0)
        small_df = small_df.copy()
        small_df.columns = sample_ids

        if REMOVED_SAMPLES:
            small_df = small_df[[c for c in small_df.columns if c not in REMOVED_SAMPLES]]

        for sample_id in small_df.columns:
            if sample_id not in sample_series:
                sample_series[sample_id] = small_df[sample_id]

    abundance_df = pd.DataFrame(sample_series).fillna(0).round().astype(int)
    log.info("Built master abundance matrix: %d proteins × %d samples", *abundance_df.shape)
    return abundance_df


# ─── STEP 2: BUILD POOLED GROUPS ────────────────────────────────────────────────

def build_pooled_groups(
    counts: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """
    For every (species, factor_of_interest) pair, pool sample IDs across all
    levels of the two remaining (non-focal) factors, and group them by the
    factor-of-interest's level.

    Returns: { (species, factor): { level: [sample_id, ...], ... }, ... }
    """
    common = counts.columns.intersection(meta_df.index)
    info = meta_df.loc[common]

    species_list = sorted(info[SPECIES_COL].dropna().unique())
    pooled: dict[tuple[str, str], dict[str, list[str]]] = {}

    for species in species_list:
        species_info = info[info[SPECIES_COL] == species]

        for factor in MARGINAL_FACTORS:
            subset = species_info
            if factor in EXCLUDE_RECOVERY_FOR_FACTORS:
                before = len(subset)
                subset = subset[subset[TEMPERATURE_COL] != RECOVERY_LEVEL]
                dropped = before - len(subset)
                if dropped:
                    log.info(
                        "  [%s | %s] excluded %d recovery-condition sample(s) prior to pooling",
                        species, factor, dropped,
                    )

            levels: dict[str, list[str]] = {}
            for level, grp in subset.groupby(factor):
                sample_ids = [s for s in grp.index if s in counts.columns]
                if sample_ids:
                    levels[level] = sample_ids

            if len(levels) >= 2:
                pooled[(species, factor)] = levels
            else:
                log.info(
                    "  [%s | %s] skipped — fewer than 2 pooled levels available",
                    species, factor,
                )

    return pooled


def save_pooled_groups_json(
    pooled_groups: dict[tuple[str, str], dict[str, list[str]]],
) -> None:
    """One groups JSON per factor, written to MARGINAL_DIR (see marginal_de_pipeline.py for format)."""
    import json

    os.makedirs(MARGINAL_DIR, exist_ok=True)

    factors = sorted({factor for (_, factor) in pooled_groups})
    for factor in factors:
        flat: dict[str, list[str]] = {}
        for (species, f), levels in pooled_groups.items():
            if f != factor:
                continue
            for level, sample_ids in levels.items():
                flat[f"{species} | {level}"] = sorted(sample_ids)

        path = os.path.join(MARGINAL_DIR, f"{factor}_groups.json")
        with open(path, "w") as fh:
            json.dump(flat, fh, indent=2, ensure_ascii=False)
        log.info("Saved pooled groupings for %s (%d groups) → %s", factor, len(flat), path)


# ─── STEP 3: RUN DESEQ PER POOLED CONTRAST ─────────────────────────────────────

def run_pooled_contrast(
    counts: pd.DataFrame,
    meta_df: pd.DataFrame,
    species: str,
    factor: str,
    level_a: str,
    level_b: str,
    comparison_name: str,
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """
    Run one pooled pairwise DE contrast (level_a vs level_b of `factor`,
    within `species`), mirroring de_proteomics_pipeline.py's run_deseq logic
    (prefilter, replicate check, categorical enforcement, fit sequence,
    named-level contrast, significance filtering) as closely as possible.
    """
    print(f"\nRunning: {comparison_name}")

    sample_ids = [
        s for s in meta_df.index
        if meta_df.loc[s, SPECIES_COL] == species
        and meta_df.loc[s, factor] in (level_a, level_b)
        and s in counts.columns
    ]
    # Re-derive from the actual pooled group lists is more precise, but this
    # mirrors the original script's straightforward metadata-driven selection.

    meta_sub = meta_df.loc[sample_ids].copy()
    counts_sub = counts[sample_ids].copy()

    print("[DEBUG] counts shape:", counts_sub.shape)
    print("[DEBUG] meta shape:", meta_sub.shape)

    # ── prefilter low-abundance proteins ───────────────────────────────────
    counts_sub = counts_sub[counts_sub.sum(axis=1) >= MIN_PROTEIN_SUM]
    if counts_sub.shape[0] == 0:
        print(f"Skipping {comparison_name}: no proteins after filtering")
        return None
    print(f"Proteins after filtering: {counts_sub.shape[0]}")

    # ── replicate check ─────────────────────────────────────────────────────
    group_counts = meta_sub[factor].value_counts()
    print("\nReplicates per group:")
    print(group_counts)
    if (group_counts < 2).any():
        print(f"Skipping {comparison_name}: insufficient replicates")
        return None

    meta_sub[factor] = meta_sub[factor].astype("category")
    meta_sub = meta_sub.loc[counts_sub.columns]

    assert all(counts_sub.columns == meta_sub.index), "ERROR: counts and metadata are misaligned"

    # ── build & fit ──────────────────────────────────────────────────────────
    dds = DeseqDataSet(
        counts=counts_sub.T,
        metadata=meta_sub,
        design=f"~{factor}",
    )

    print("Fitting DESeq2 model...")
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.refit_cooks = False
    dds.fit_LFC()

    contrast = [factor, level_b, level_a]
    print("Contrast:", contrast)

    ds = DeseqStats(
        dds,
        contrast=contrast,
        alpha=P_VALUE_THRESHOLD,
        cooks_filter=False,
        independent_filter=False,
    )

    print("Running Wald test...")
    ds.run_wald_test()
    ds._p_value_adjustment()
    ds.summary()

    results = ds.results_df.copy()
    lfc = results["log2FoldChange"]
    padj = results["padj"]
    pval = results["pvalue"]

    valid = (~lfc.isna()) & (~padj.isna())
    significant = valid & (padj <= P_VALUE_THRESHOLD)
    mask = (~significant) | (lfc.abs() <= LFC_THRESHOLD)

    filtered_lfc = lfc.mask(mask)
    filtered_padj = padj.mask(mask)

    print("[DEBUG] Significant proteins:", significant.sum())
    print("[DEBUG] Final DE proteins:", (~mask).sum())

    return filtered_lfc, filtered_padj, pval


# ─── STEP 4: SAVE RESULTS ───────────────────────────────────────────────────────

def save_results(
    lfc_map: dict,
    padj_map: dict,
    pval_map: dict,
) -> None:
    """One three-sheet Excel file per factor of interest, written to MARGINAL_DIR."""
    os.makedirs(MARGINAL_DIR, exist_ok=True)

    def key_to_label(key: tuple) -> str:
        species, factor, level_a, level_b = key
        return f"{species} | {level_a} vs {level_b}"

    def to_df(data: dict, factor: str) -> pd.DataFrame:
        renamed = {key_to_label(k): v for k, v in data.items() if k[1] == factor}
        return pd.DataFrame(renamed).reset_index().rename(columns={"index": "protein"})

    factors = sorted({key[1] for key in lfc_map})

    for factor in factors:
        lfc_df = to_df(lfc_map, factor)
        padj_df = to_df(padj_map, factor)
        pval_df = to_df(pval_map, factor)

        print(f"\n[{factor}] LFC df shape: {lfc_df.shape}")
        print(f"[{factor}] Padj df shape: {padj_df.shape}")
        print(f"[{factor}] Pval df shape: {pval_df.shape}")

        output_path = os.path.join(MARGINAL_DIR, f"{factor}_de_raw_results.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            lfc_df.to_excel(writer, sheet_name="log2FoldChange", index=False)
            padj_df.to_excel(writer, sheet_name="padj_values", index=False)
            pval_df.to_excel(writer, sheet_name="p_values", index=False)

        print(f"Saved → {output_path}")

    notification.notify(title="DESeq2", message="Marginal proteomics DE run finished!")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def pipeline(groups_only: bool = False) -> None:
    meta_df = load_sample_metadata()
    abundance_df = build_master_abundance()

    pooled_groups = build_pooled_groups(abundance_df, meta_df)
    save_pooled_groups_json(pooled_groups)

    if groups_only:
        print(f"\ngroups-only mode: skipping DESeq2 fitting. Inspect the *_groups.json files in {MARGINAL_DIR}, then re-run without --groups-only.")
        return

    lfc_map, padj_map, pval_map = {}, {}, {}

    for (species, factor), levels in pooled_groups.items():
        level_names = sorted(levels.keys())
        for level_a, level_b in combinations(level_names, 2):
            comparison_name = f"{species} | {factor}: {level_a} vs {level_b}"
            try:
                result = run_pooled_contrast(
                    abundance_df, meta_df, species, factor, level_a, level_b, comparison_name,
                )
            except Exception as e:
                print(f"\nERROR in {comparison_name}")
                print(str(e))
                continue

            if result is None:
                continue

            key = (species, factor, level_a, level_b)
            lfc_map[key], padj_map[key], pval_map[key] = result

    save_results(lfc_map, padj_map, pval_map)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pipeline(groups_only="--groups-only" in sys.argv)