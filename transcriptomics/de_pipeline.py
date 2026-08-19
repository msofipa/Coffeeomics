"""
Differential Expression pipeline — Coffeomics transcriptomics.

Contrast groups are loaded from contrast_groups.json.
Expected JSON format — a list of string pairs, each a pipe-separated label:

    [
        ["C. arabica | 25/20°C | 380ppm | without",
         "C. arabica | 25/20°C | 380ppm | medium"],
        ...
    ]

Each label encodes: Species | Temperature | CO2_ppm | Water_Stress
(in the same order as ATTR_COLS below).
Only pairs where exactly one attribute differs are processed;
all others are skipped with a warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from plyer import notification
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\papan\Desktop\Coffeomics")
TRANS_DIR     = BASE_DIR / "transcriptomics"
INPUT_DATA    = TRANS_DIR / "trans_normalizations" / "trans_150bins_window5_normalized_counts.csv"
SAMPLE_INFO   = BASE_DIR / "sample_information.csv"
CONTRAST_JSON = BASE_DIR / "proteomics" / "contrast_groups.json"
OUTPUT_DIR    = TRANS_DIR / "de_raw_results"
OUTPUT_FILE   = "trans_counts_150bins_window5_lfc1_de_raw_results.xlsx"

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
REMOVED_SAMPLE    = "40A"
P_VALUE_THRESHOLD = 0.01
LFC_THRESHOLD     = 1.0
MIN_SAMPLES       = 3

# ─── METADATA STRUCTURE ───────────────────────────────────────────────────────
# Column names in sample_information.csv, in the same order as the
# pipe-separated label in contrast_groups.json.
ATTR_COLS       = ["Species", "Temperature", "CO2_ppm", "Water_Stress"]
SEP             = " | "
# Attributes to never use as the DE design variable
SKIP_ATTRIBUTES = {"Species"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def parse_label(label: str) -> dict[str, str]:
    """'C. arabica | 25/20°C | 380ppm | without' → {'Species': ..., ...}"""
    parts = [p.strip() for p in label.split(SEP)]
    if len(parts) != len(ATTR_COLS):
        raise ValueError(
            f"Label has {len(parts)} fields, expected {len(ATTR_COLS)}: {label!r}"
        )
    return dict(zip(ATTR_COLS, parts))


def varying_attribute(lbl1: dict[str, str], lbl2: dict[str, str]) -> str | None:
    """Return the single attribute that differs between two labels; None otherwise."""
    diffs = [k for k in ATTR_COLS if lbl1[k] != lbl2[k]]
    return diffs[0] if len(diffs) == 1 else None



# ─── PIPELINE STEPS ───────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]]]:
    """Load count matrix, sample metadata, and pre-computed contrast groups."""
    counts = (
        pd.read_csv(INPUT_DATA, index_col=0)
        .drop(columns=[REMOVED_SAMPLE], errors="ignore")
    )
    sample_info = pd.read_csv(SAMPLE_INFO, header=0, index_col="Sample_ID")
    with open(CONTRAST_JSON) as f:
        contrast_groups: list[list[str]] = json.load(f)

    log.info(
        "Loaded: %d genes × %d samples | %d contrast groups",
        *counts.shape,
        len(contrast_groups),
    )
    return counts, sample_info, contrast_groups


def prepare_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN → 0, round and cast to int (DESeq2 requires integer counts)."""
    return df.fillna(0).round().astype(int)


def build_dds_objects(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    contrast_groups: dict[str, list[str]],
) -> dict[tuple, DeseqDataSet]:
    """
    Build one DeseqDataSet per valid contrast pair.

    contrast_groups is a dict: { condition_label → [sample_id, ...] }
    e.g. { "C. arabica | 25/20°C | 380ppm | without": ["7A", "7B", "7C"], ... }

    All pairwise combinations of the 36 labels are generated here;
    only pairs where exactly one attribute differs proceed to DDS construction.

    Validity requirements:
    - Exactly one attribute differs between the two condition labels.
    - The varying attribute is not in SKIP_ATTRIBUTES.
    - At least 2 distinct biological samples (unique prefix before replicate suffix).
    - Both levels of the varying attribute are present in sample_info.
    - At least MIN_SAMPLES columns available in the count matrix.
    """
    from itertools import combinations as _combinations

    labels    = list(contrast_groups.keys())
    all_pairs = list(_combinations(labels, 2))
    log.info("%d condition labels → %d candidate pairs.", len(labels), len(all_pairs))

    # Align metadata to columns present in the count matrix
    common      = counts.columns.intersection(sample_info.index)
    sample_info = sample_info.loc[common].reindex(counts.columns)

    dds_list: dict[tuple, DeseqDataSet] = {}
    skipped = 0

    for label1, label2 in all_pairs:
        lbl1 = parse_label(label1)
        lbl2 = parse_label(label2)

        # ── guard: exactly one attribute must differ ──────────────────────────
        attr = varying_attribute(lbl1, lbl2)
        if attr is None or attr in SKIP_ATTRIBUTES:
            skipped += 1
            continue

        # ── sample IDs come directly from the JSON — no metadata lookup needed ─
        all_ids  = contrast_groups[label1] + contrast_groups[label2]
        available = [s for s in all_ids if s in counts.columns]

        # ── guard: ≥2 distinct biological samples ─────────────────────────────
        if len({s[:-1] for s in available}) < 2:
            skipped += 1
            continue

        meta = sample_info.loc[[s for s in available if s in sample_info.index]]

        # ── guard: both levels of the design variable present in metadata ──────
        if meta[attr].nunique() < 2:
            skipped += 1
            continue

        # ── guard: ≥ MIN_SAMPLES columns in count matrix ──────────────────────
        counts_sub = counts[[s for s in available if s in counts.columns]]
        if counts_sub.shape[1] < MIN_SAMPLES:
            skipped += 1
            continue

        dds = DeseqDataSet(
            counts=counts_sub.T,
            metadata=meta.loc[counts_sub.columns],
            design=f"~{attr}",
        )

        # Key: (val1, val2, fixed_last, ..., fixed_first) — same order as original
        fixed_vals = [lbl1[k] for k in ATTR_COLS if k != attr]
        key = (lbl1[attr], lbl2[attr], *reversed(fixed_vals))
        dds_list[key] = dds

    log.info("DDS objects built: %d valid | %d skipped", len(dds_list), skipped)
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

    Returns three dicts keyed by contrast tuple:
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

        # Mask: keep only genes that are significant AND exceed LFC threshold
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
    """Assemble result DataFrames and write a three-sheet Excel file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE

    def to_df(data: dict) -> pd.DataFrame:
        return pd.DataFrame(data).reset_index().rename(columns={"index": "gene"})

    lfc_df  = to_df(lfc_map)
    padj_df = to_df(padj_map)
    pval_df = to_df(pval_map)

    log.info(
        "Result shapes — LFC: %s | padj: %s | pval: %s",
        lfc_df.shape, padj_df.shape, pval_df.shape,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        lfc_df.to_excel(writer,  sheet_name="log2FoldChange", index=True)
        padj_df.to_excel(writer, sheet_name="padj_values",    index=True)
        pval_df.to_excel(writer, sheet_name="p_values",       index=True)

    log.info("Saved → %s", output_path)
    notification.notify(message="DE run finished!")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def pipeline() -> None:
    counts_raw, sample_info, contrast_groups = load_inputs()
    counts = prepare_counts(counts_raw)

    dds_list = build_dds_objects(counts, sample_info, contrast_groups)
    fit_models(dds_list)

    ds_list  = build_stat_objects(dds_list)
    lfc_map, padj_map, pval_map = collect_results(ds_list)

    save_results(lfc_map, padj_map, pval_map)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline()
