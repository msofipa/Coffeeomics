"""
Marginal (pooled) Differential Expression pipeline — Coffeomics transcriptomics.

Complementary to de_pipeline.py's fully-factorial pairwise contrasts, in which
every unique combination of Species | Temperature | CO2_ppm | Water_Stress is
treated as its own group. Here, for a given factor of interest (Temperature,
CO2_ppm, or Water_Stress), samples are pooled across all levels of the two
remaining (non-focal) factors, WITHIN each species — yielding one aggregated
sample group per (species, factor level). Pairwise DE contrasts are then run
between these pooled groups, same species, same factor, using the identical
PyDESeq2 workflow and significance criteria as de_pipeline.py.

This increases replication (and thus power) per group for detecting each
factor's main effect, at the cost of averaging over any interaction between
that factor and the other two.

Groups are derived directly from sample_information.csv (NOT from
contrast_groups.json, which only encodes the fully-factorial 36 labels).

Recovery ("Rec") is a level of Temperature (post-stress recovery timepoint),
not of Water_Stress. Accordingly, recovery-condition samples are excluded
prior to pooling specifically for the "temperature-collapsed" analyses —
i.e. whenever Temperature is one of the two non-focal factors being pooled
away (factor of interest = CO2_ppm or Water_Stress). When Temperature itself
is the factor of interest, "Rec" is a real level being contrasted and is
never excluded. See RECOVERY_LEVEL / EXCLUDE_RECOVERY_FOR_FACTORS below —
verify the exact "Rec" label against sample_information.csv.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from plyer import notification
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\papan\Desktop\Coffeeomics")
TRANS_DIR     = BASE_DIR / "transcriptomics"
INPUT_DATA    = TRANS_DIR / "normalizations" / "trans_pydeseq2_normalized_counts.csv"
SAMPLE_INFO   = BASE_DIR / "sample_information.csv"

# All marginal-analysis output (per-factor groupings JSON + per-factor DE
# results Excel files) is written into this one folder, one pair of files
# per factor of interest.
MARGINAL_DIR  = TRANS_DIR / "de_raw_results" / "marginal"

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
REMOVED_SAMPLE    = "40A"
P_VALUE_THRESHOLD = 0.01
LFC_THRESHOLD     = 1.0
MIN_SAMPLES       = 3

# ─── METADATA STRUCTURE ───────────────────────────────────────────────────────
ATTR_COLS = ["Species", "Temperature", "CO2_ppm", "Water_Stress"]
SPECIES_COL = "Species"

# Factors eligible to be the "factor of interest" for a marginal analysis.
# Species is never a factor of interest — pooling always happens within species.
MARGINAL_FACTORS = ["Temperature", "CO2_ppm", "Water_Stress"]

# Exact label used for the post-stress recovery level within Temperature.
# *** VERIFY against sample_information.csv — adjust to match exactly. ***
RECOVERY_LEVEL = "Rec"
TEMPERATURE_COL = "Temperature"

# Factors of interest for which recovery-condition samples are excluded prior
# to pooling — i.e. whenever Temperature (which contains the "Rec" level) is
# one of the two non-focal factors being pooled/collapsed away. This is the
# "temperature-collapsed analysis": factor of interest ≠ Temperature.
EXCLUDE_RECOVERY_FOR_FACTORS = {"CO2_ppm", "Water_Stress"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def biological_replicate_count(sample_ids: list[str]) -> int:
    """Number of distinct biological samples (unique prefix before replicate suffix)."""
    return len({s[:-1] for s in sample_ids})


# ─── PIPELINE STEPS ───────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load count matrix and sample metadata (no contrast_groups.json needed here)."""
    counts = (
        pd.read_csv(INPUT_DATA, index_col=0)
        .drop(columns=[REMOVED_SAMPLE], errors="ignore")
    )
    sample_info = pd.read_csv(SAMPLE_INFO, header=0, index_col="Sample_ID")

    log.info("Loaded: %d genes × %d samples | %d metadata rows",
              *counts.shape, len(sample_info))
    return counts, sample_info


def prepare_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN → 0, round and cast to int (DESeq2 requires integer counts)."""
    return df.fillna(0).round().astype(int)


def build_pooled_groups(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """
    For every (species, factor_of_interest) pair, pool sample IDs across all
    levels of the two remaining (non-focal) factors, and group them by the
    factor-of-interest's level.

    Returns: { (species, factor): { level: [sample_id, ...], ... }, ... }
    """
    # Align metadata to columns present in the count matrix
    common = counts.columns.intersection(sample_info.index)
    info = sample_info.loc[common]

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
    """
    Dump the pooled groupings to JSON for inspection/audit before (or instead
    of) running the full DESeq2 pipeline — one file per factor of interest,
    written to MARGINAL_DIR.

    Each file (e.g. "CO2_ppm_groups.json") is a flat dict, one entry per
    pooled group for that factor, across all species:
        { "<species> | <level>": [sample_id, ...], ... }

    e.g. in "CO2_ppm_groups.json": "C. arabica | 380ppm": ["7A", "7B", ...]
    (pooled across every Temperature × Water_Stress combination for that
    species and CO2 level, minus any recovery-condition exclusions applied).
    """
    MARGINAL_DIR.mkdir(parents=True, exist_ok=True)

    factors = sorted({factor for (_, factor) in pooled_groups})
    for factor in factors:
        flat: dict[str, list[str]] = {}
        for (species, f), levels in pooled_groups.items():
            if f != factor:
                continue
            for level, sample_ids in levels.items():
                flat[f"{species} | {level}"] = sorted(sample_ids)

        path = MARGINAL_DIR / f"{factor}_groups.json"
        with open(path, "w") as fh:
            json.dump(flat, fh, indent=2, ensure_ascii=False)
        log.info("Saved pooled groupings for %s (%d groups) → %s", factor, len(flat), path)


def build_dds_objects(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    pooled_groups: dict[tuple[str, str], dict[str, list[str]]],
) -> dict[tuple, DeseqDataSet]:
    """
    Build one DeseqDataSet per valid (species, factor, level_a, level_b) pooled
    contrast, using design=~{factor}.

    Validity requirements (mirrors de_pipeline.py's guards):
    - At least 2 distinct biological samples (unique prefix before replicate suffix).
    - Both levels of the factor of interest present in the pooled metadata.
    - At least MIN_SAMPLES columns available in the count matrix.
    """
    common = counts.columns.intersection(sample_info.index)
    info = sample_info.loc[common].reindex(counts.columns)

    dds_list: dict[tuple, DeseqDataSet] = {}
    skipped = 0

    for (species, factor), levels in pooled_groups.items():
        level_names = sorted(levels.keys())
        for level_a, level_b in combinations(level_names, 2):
            available = [
                s for s in (levels[level_a] + levels[level_b])
                if s in counts.columns
            ]

            # ── guard: ≥2 distinct biological samples ─────────────────────────
            if biological_replicate_count(available) < 2:
                skipped += 1
                continue

            meta = info.loc[[s for s in available if s in info.index]]

            # ── guard: both levels of the design variable present ─────────────
            if meta[factor].nunique() < 2:
                skipped += 1
                continue

            # ── guard: ≥ MIN_SAMPLES columns in count matrix ──────────────────
            counts_sub = counts[[s for s in available if s in counts.columns]]
            if counts_sub.shape[1] < MIN_SAMPLES:
                skipped += 1
                continue

            dds = DeseqDataSet(
                counts=counts_sub.T,
                metadata=meta.loc[counts_sub.columns],
                design=f"~{factor}",
            )

            key = (species, factor, level_a, level_b)
            dds_list[key] = dds

    log.info("Pooled DDS objects built: %d valid | %d skipped", len(dds_list), skipped)
    return dds_list


def fit_models(dds_list: dict[tuple, DeseqDataSet]) -> None:
    """Fit the full DESeq2 dispersion and LFC model for every dataset."""
    n = len(dds_list)
    for i, dds in enumerate(dds_list.values(), 1):
        log.info("  Fitting model %d / %d …", i, n)
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        dds.refit_cooks = False
        dds.fit_LFC()


def build_stat_objects(
    dds_list: dict[tuple, DeseqDataSet],
) -> dict[tuple, DeseqStats]:
    """Wrap every fitted DDS in a DeseqStats object ready for Wald testing."""
    return {
        key: DeseqStats(
            dds,
            contrast=np.array([0, 1]),
            alpha=P_VALUE_THRESHOLD,
            cooks_filter=False,
            independent_filter=True,
        )
        for key, dds in dds_list.items()
    }


def collect_results(
    ds_list: dict[tuple, DeseqStats],
) -> tuple[dict, dict, dict]:
    """
    Run Wald test on every DeseqStats object and apply significance filters.

    Returns three dicts keyed by (species, factor, level_a, level_b):
      lfc_map  — LFC values for genes passing padj ≤ threshold AND |LFC| > threshold
      padj_map — matching adjusted p-values (same mask)
      pval_map — raw p-values for ALL genes (unfiltered)
    """
    lfc_map, padj_map, pval_map = {}, {}, {}
    n = len(ds_list)

    for i, (key, ds) in enumerate(ds_list.items(), 1):
        log.info("  Wald test %d / %d: %s", i, n, key)

        ds.run_wald_test()
        ds._p_value_adjustment()
        ds.summary()

        res  = ds.results_df.copy()
        lfc  = res["log2FoldChange"]
        padj = res["padj"]

        valid       = lfc.notna() & padj.notna()
        significant = valid & (padj <= P_VALUE_THRESHOLD)
        mask        = ~significant | (lfc.abs() <= LFC_THRESHOLD)

        lfc_map[key]  = lfc.mask(mask)
        padj_map[key] = padj.mask(mask)
        pval_map[key] = res["pvalue"]  # raw, unfiltered

    return lfc_map, padj_map, pval_map


def save_results(
    lfc_map: dict,
    padj_map: dict,
    pval_map: dict,
) -> None:
    """
    Assemble result DataFrames and write one three-sheet Excel file per
    factor of interest into MARGINAL_DIR (e.g. "CO2_ppm_de_raw_results.xlsx"),
    each covering every species' pairwise contrasts for that factor.
    """
    MARGINAL_DIR.mkdir(parents=True, exist_ok=True)

    def key_to_label(key: tuple) -> str:
        species, factor, level_a, level_b = key
        return f"{species} | {level_a} vs {level_b}"

    def to_df(data: dict, factor: str) -> pd.DataFrame:
        renamed = {key_to_label(k): v for k, v in data.items() if k[1] == factor}
        return pd.DataFrame(renamed).reset_index().rename(columns={"index": "gene"})

    factors = sorted({key[1] for key in lfc_map})

    for factor in factors:
        lfc_df  = to_df(lfc_map, factor)
        padj_df = to_df(padj_map, factor)
        pval_df = to_df(pval_map, factor)

        log.info(
            "[%s] result shapes — LFC: %s | padj: %s | pval: %s",
            factor, lfc_df.shape, padj_df.shape, pval_df.shape,
        )

        output_path = MARGINAL_DIR / f"{factor}_de_raw_results.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            lfc_df.to_excel(writer,  sheet_name="log2FoldChange", index=True)
            padj_df.to_excel(writer, sheet_name="padj_values",    index=True)
            pval_df.to_excel(writer, sheet_name="p_values",       index=True)

        log.info("Saved → %s", output_path)

    notification.notify(message="Marginal DE run finished!")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def pipeline(groups_only: bool = False) -> None:
    counts_raw, sample_info = load_inputs()
    counts = prepare_counts(counts_raw)

    pooled_groups = build_pooled_groups(counts, sample_info)
    save_pooled_groups_json(pooled_groups)

    if groups_only:
        log.info("groups-only mode: skipping DESeq2 fitting. Inspect the *_groups.json files in %s, then re-run without --groups-only.", MARGINAL_DIR)
        return

    dds_list = build_dds_objects(counts, sample_info, pooled_groups)
    fit_models(dds_list)

    ds_list  = build_stat_objects(dds_list)
    lfc_map, padj_map, pval_map = collect_results(ds_list)

    save_results(lfc_map, padj_map, pval_map)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pipeline(groups_only="--groups-only" in sys.argv)