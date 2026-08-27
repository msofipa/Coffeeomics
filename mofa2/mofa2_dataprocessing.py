"""
build_mofa_input.py
--------------------
Assembles transcriptomics, proteomics, and metabolomics matrices from the
Coffeeomics project into the long-format DataFrame that mofapy2 expects:

    columns: sample, group, feature, view, value

Design choices tied to the project's known conventions:
  - Paths are anchored absolutely with pathlib.Path (per project convention;
    avoids working-directory-dependent FileNotFoundErrors).
  - Encoding: tries utf-8 first, falls back to cp1252 (Windows-generated
    files with degree symbols in condition strings).
  - Sample metadata (species, temperature, CO2, water stress) is derived by
    INVERTING each layer's own contrast_groups JSON, since those files map
    condition-string -> [sample IDs], not sample -> condition. Contrast
    groups are layer-specific (different physical samples/replicates went
    into transcriptomics vs. proteomics vs. metabolomics), so each view is
    matched against its OWN contrast groups file, not a shared one -- a
    sample ID from one layer has no guaranteed relationship to the same-
    looking ID in another layer.
  - Handles the known 5-level MultiIndex header convention
    (Species, Temperature, CO2, Water Stress, Sample ID) via
    flatten_multiindex_to_sample_id(), but auto-detects and skips this if a
    given input file already has flat sample-ID columns (e.g. matrices
    already prepared for PyDESeq2, which needs flat colData).
  - Emits a diagnostic report of samples present in each omics matrix but
    absent from contrast_groups.json (and vice versa), directly relevant to
    the open question about ~52 unmatched metabolomics samples.

>>> EDIT THE "CONFIG" SECTION BELOW BEFORE RUNNING <<<
This script cannot see your actual files, so the paths, sheet names, and
"already flat vs. MultiIndex" assumptions below are best guesses based on
the project's established conventions. Adjust as needed -- the diagnostic
output at the bottom will tell you quickly if a loader guessed wrong
(e.g. feature/sample counts that look transposed, or near-total sample
mismatch against contrast_groups.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =========================================================================
# CONFIG -- edit these paths for your machine
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

# Contrast groups are layer-specific -- each omics layer has its own file
# mapping condition strings to that layer's sample IDs.
CONTRAST_GROUPS_TRANSCRIPTOMICS_PATH = BASE_DIR / "transcriptomics" / "contrast_groups_transcriptomics.json"
CONTRAST_GROUPS_PROTEOMICS_PATH = BASE_DIR / "proteomics" / "contrast_groups_proteomics.json"
CONTRAST_GROUPS_METABOLOMICS_PATH = BASE_DIR / "metabolomics" / "contrast_groups_metabolomics.json"

# Adjust these to the actual normalized/filtered matrices you want MOFA to see.
# Expectation: features (genes/proteins/metabolites) as rows, samples as columns.
TRANSCRIPTOMICS_PATH = BASE_DIR / "transcriptomics" / "normalizations" / "trans_pydeseq2_normalized_counts.csv"
PROTEOMICS_PATH = BASE_DIR / "proteomics" / "data" / "prot_normalized_pydeseq2_manual_50_above_10.xlsx"
METABOLOMICS_PATH = BASE_DIR / "metabolomics" / "data" / "metabolomics_counts.csv"

OUTPUT_LONG_DF_PATH = BASE_DIR / "integration" / "mofa2" / "mofa_input_long.csv"

# Feature selection for high-dimensional views (transcriptomics/proteomics).
# Metabolomics (50 features) is left untouched.
N_TOP_VARIABLE_TRANSCRIPTS = 5000
N_TOP_VARIABLE_PROTEINS = 5000

# --- Known, manually-confirmed corrections to the metabolomics matrix ---
# These come from the user's own verification against raw sample records,
# NOT from automated inference. Keep this list current if the underlying
# data files are fixed at the source (at which point these lines can be
# deleted).
#   - "10D" is a mislabeled column. It is NOT a valid sample ID for this
#     plant per sample_information.csv (only 10A/10B/10C/10E exist), and its
#     values differ clearly from the native "10E" column (ruling out
#     "duplicate export of 10E" as the explanation). "10B" is the one
#     sample persistently missing from the metabolomics matrix across every
#     diagnostic run -- "10D" is almost certainly that mislabeled column.
#   - "29D" does not correspond to a real sample and should be dropped.
METABOLOMICS_SAMPLE_ID_RENAMES = {"10D": "10B"}
METABOLOMICS_SAMPLES_TO_DROP = ["29D"]
 
# Samples confirmed to exist in ALL THREE layers but currently absent from
# the metabolomics matrix. This is a genuine data gap (not a naming issue)
# -- these should be recovered/investigated upstream. The script only
# reports on this; it cannot fabricate the missing values.
EXPECTED_IN_ALL_LAYERS_BUT_MISSING_FROM_METABOLOMICS = ["28C"]

# =========================================================================
# I/O helpers
# =========================================================================


def read_table_flex(path: Path, header: Optional[list] = None) -> pd.DataFrame:
    """Read a csv/tsv/xlsx table with utf-8-first, cp1252-fallback encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        # openpyxl doesn't need an encoding kwarg
        df = pd.read_excel(path, header=header if header is not None else 0, index_col=0)
        _print_header_sanity_check(df, path)
        return df

    for enc in ("utf-8", "cp1252"):
        try:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, encoding=enc,
                              header=header if header is not None else 0, index_col=0)
            _print_header_sanity_check(df, path)
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with utf-8 or cp1252")


def _print_header_sanity_check(df: pd.DataFrame, path: Path) -> None:
    """
    Print what was actually parsed as columns/index, so a wrong header= or
    index_col= assumption is caught immediately rather than silently
    producing a garbage matrix (e.g. feature names ending up as columns,
    or a metadata row being read as a sample ID).
    """
    cols_preview = list(df.columns[:5])
    print(f"  [check] {path.name}: index name={df.index.name!r}, "
          f"first index values={list(df.index[:3])}, "
          f"first columns={cols_preview}")


def flatten_multiindex_to_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a 5-level MultiIndex column header
    (Species, Temperature, CO2, Water Stress, Sample ID) down to just the
    Sample ID level, matching the project's established
    flatten_multiindex_columns_to_sample_id() convention.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Sample ID is the last level per project convention
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df


def looks_like_flat_sample_columns(df: pd.DataFrame) -> bool:
    """Heuristic: already-flat columns won't be a MultiIndex."""
    return not isinstance(df.columns, pd.MultiIndex)


# =========================================================================
# Metadata: invert contrast_groups.json into sample -> condition lookup
# =========================================================================


def load_sample_metadata(contrast_groups_path: Path) -> pd.DataFrame:
    """
    contrast_groups.json maps condition strings like
        "C. arabica | 25/20°C | 380ppm | medium" -> [sample_id, ...]
    Invert this into a per-sample metadata table:
        sample, species, temperature, co2, water_stress
    """
    with open(contrast_groups_path, "r", encoding="utf-8") as f:
        contrast_groups = json.load(f)

    records = []
    for condition_str, sample_ids in contrast_groups.items():
        parts = [p.strip() for p in condition_str.split("|")]
        if len(parts) != 4:
            print(f"  [warn] unexpected condition string format, skipping: {condition_str!r}",
                  file=sys.stderr)
            continue
        species, temperature, co2, water_stress = parts
        for sid in sample_ids:
            records.append({
                "sample": sid,
                "species": species,
                "temperature": temperature,
                "co2": co2,
                "water_stress": water_stress,
            })

    meta = pd.DataFrame(records)
    if meta["sample"].duplicated().any():
        dupes = meta.loc[meta["sample"].duplicated(), "sample"].tolist()
        print(f"  [warn] {len(dupes)} sample IDs appear in more than one condition "
              f"in contrast_groups.json: {dupes[:10]}{'...' if len(dupes) > 10 else ''}",
              file=sys.stderr)
        meta = meta.drop_duplicates(subset="sample", keep="first")

    return meta.set_index("sample")


# =========================================================================
# Per-view loaders -> long format
# =========================================================================


def select_top_variable_features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """df: features x samples. Keep the n highest-variance features."""
    if df.shape[0] <= n:
        return df.copy()
    variances = df.var(axis=1, skipna=True)
    top_features = variances.sort_values(ascending=False).head(n).index
    return df.loc[top_features].copy()


def wide_to_long(df: pd.DataFrame, view_name: str) -> pd.DataFrame:
    """df: features (index) x samples (columns) -> long df with a view column."""
    long_df = df.reset_index().melt(id_vars=df.index.name or "index",
                                     var_name="sample", value_name="value")
    long_df = long_df.rename(columns={df.index.name or "index": "feature"})
    long_df["view"] = view_name
    return long_df


def attach_group(long_df: pd.DataFrame, meta: pd.DataFrame, view_name: str) -> pd.DataFrame:
    """Merge a view's long-format data with THAT VIEW's own metadata table."""
    merged = long_df.merge(meta[["species"]], left_on="sample", right_index=True, how="left")
    merged = merged.rename(columns={"species": "group"})
    n_missing = merged["group"].isna().sum()
    if n_missing:
        n_samples_missing = merged.loc[merged["group"].isna(), "sample"].nunique()
        print(f"  [warn] {view_name}: {n_samples_missing} sample(s) "
              f"({n_missing} rows) not found in this layer's contrast groups file "
              f"-- will be dropped before MOFA input.")
    return merged


def load_transcriptomics(path: Path, meta: pd.DataFrame) -> pd.DataFrame:
    print(f"Loading transcriptomics from {path} ...")
    df = read_table_flex(path)
    if not looks_like_flat_sample_columns(df):
        df = flatten_multiindex_to_sample_id(df)
    df = select_top_variable_features(df, N_TOP_VARIABLE_TRANSCRIPTS)
    df.index.name = "feature"
    print(f"  -> {df.shape[0]} features x {df.shape[1]} samples after feature selection")
    long_df = wide_to_long(df, "transcriptomics")
    return attach_group(long_df, meta, "transcriptomics")


def load_proteomics(path: Path, meta: pd.DataFrame) -> pd.DataFrame:
    print(f"Loading proteomics from {path} ...")
    df = read_table_flex(path)
    if not looks_like_flat_sample_columns(df):
        df = flatten_multiindex_to_sample_id(df)
    df = select_top_variable_features(df, N_TOP_VARIABLE_PROTEINS)
    df.index.name = "feature"
    print(f"  -> {df.shape[0]} features x {df.shape[1]} samples after feature selection")
    long_df = wide_to_long(df, "proteomics")
    return attach_group(long_df, meta, "proteomics")


def apply_metabolomics_id_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply known, manually-confirmed sample ID corrections to the
    metabolomics matrix (see CONFIG). These are one-off fixes for specific
    mislabeled/spurious columns identified by manual review, not a general
    cleaning routine -- remove this call once the source file is corrected.
    """
    df = df.copy()

    to_drop = [c for c in METABOLOMICS_SAMPLES_TO_DROP if c in df.columns]
    if to_drop:
        print(f"  [correction] dropping non-existent sample column(s) from "
              f"metabolomics: {to_drop}")
        df = df.drop(columns=to_drop)

    rename_map = {old: new for old, new in METABOLOMICS_SAMPLE_ID_RENAMES.items()
                  if old in df.columns}
    if rename_map:
        print(f"  [correction] renaming metabolomics sample column(s): {rename_map}")
        df = df.rename(columns=rename_map)

    return df


def load_metabolomics(path: Path, meta: pd.DataFrame) -> pd.DataFrame:
    print(f"Loading metabolomics from {path} ...")
    # Metabolomics file is known to use the 5-level MultiIndex header
    df = read_table_flex(path, header=[0, 1, 2, 3, 4])
    df = flatten_multiindex_to_sample_id(df)
    df = apply_metabolomics_id_corrections(df)
    df.index.name = "feature"
    # Log-transform abundances (metabolomics values are typically raw intensities)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    log_df = np.log2(numeric.clip(lower=1e-6))
    print(f"  -> {log_df.shape[0]} features x {log_df.shape[1]} samples (log2-transformed)")
    long_df = wide_to_long(log_df, "metabolomics")
    return attach_group(long_df, meta, "metabolomics")


# =========================================================================
# Diagnostics: sample overlap between each view and contrast_groups.json
# =========================================================================


def report_sample_overlap(view_long_dfs: dict, view_metas: dict) -> None:
    print("\n=== Sample overlap diagnostics (each view vs. its OWN contrast groups file) ===")
    for view_name, long_df in view_long_dfs.items():
        meta_samples = set(view_metas[view_name].index)
        view_samples = set(long_df["sample"].unique())
        matched = view_samples & meta_samples
        unmatched_in_view = view_samples - meta_samples
        unmatched_in_meta = meta_samples - view_samples
        print(f"\n[{view_name}]")
        print(f"  samples in data matrix:              {len(view_samples)}")
        print(f"  samples in this layer's contrast groups file: {len(meta_samples)}")
        print(f"  matched:                              {len(matched)}")
        print(f"  in data but NOT in contrast groups:   {len(unmatched_in_view)}")
        if unmatched_in_view:
            sample_list = sorted(unmatched_in_view)
            print(f"    e.g. {sample_list[:10]}{'...' if len(sample_list) > 10 else ''}")
        print(f"  in contrast groups but NOT in data:   {len(unmatched_in_meta)}")
        if unmatched_in_meta:
            sample_list = sorted(unmatched_in_meta)
            print(f"    e.g. {sample_list[:10]}{'...' if len(sample_list) > 10 else ''}")


def report_expected_but_missing(view_long_dfs: dict) -> None:
    """
    Flag samples the user has confirmed SHOULD exist in the metabolomics
    matrix (because they exist in the other layers) but currently don't.
    Unlike the general overlap diagnostic, this is a targeted check against
    a manually-confirmed list -- a hit here means real missing data, not a
    naming mismatch, and needs upstream investigation rather than a script fix.
    """
    if "metabolomics" not in view_long_dfs:
        return
    metab_samples = set(view_long_dfs["metabolomics"]["sample"].unique())
    still_missing = [s for s in EXPECTED_IN_ALL_LAYERS_BUT_MISSING_FROM_METABOLOMICS
                      if s not in metab_samples]
    if still_missing:
        print(f"\n[ACTION NEEDED] confirmed-real sample(s) still absent from the "
              f"metabolomics matrix: {still_missing}. This is a genuine data gap "
              f"(not a naming issue) -- check raw metabolomics measurement/extraction "
              f"records for these IDs. MOFA will otherwise treat metabolomics values "
              f"for these samples as fully missing without raising an error.")


# =========================================================================
# Main assembly
# =========================================================================


def main():
    print("Loading per-layer contrast groups ...")
    view_metas = {}
    for view_name, path in [
        ("transcriptomics", CONTRAST_GROUPS_TRANSCRIPTOMICS_PATH),
        ("proteomics", CONTRAST_GROUPS_PROTEOMICS_PATH),
        ("metabolomics", CONTRAST_GROUPS_METABOLOMICS_PATH),
    ]:
        try:
            meta = load_sample_metadata(path)
            view_metas[view_name] = meta
            print(f"  [{view_name}] -> {len(meta)} samples, "
                  f"{meta['species'].nunique()} species (from {path.name})")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    view_long_dfs = {}
    if "transcriptomics" in view_metas:
        try:
            view_long_dfs["transcriptomics"] = load_transcriptomics(
                TRANSCRIPTOMICS_PATH, view_metas["transcriptomics"])
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
    if "proteomics" in view_metas:
        try:
            view_long_dfs["proteomics"] = load_proteomics(
                PROTEOMICS_PATH, view_metas["proteomics"])
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
    if "metabolomics" in view_metas:
        try:
            view_long_dfs["metabolomics"] = load_metabolomics(
                METABOLOMICS_PATH, view_metas["metabolomics"])
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    if not view_long_dfs:
        print("No views loaded successfully -- check CONFIG paths at the top of this script.")
        return

    report_sample_overlap(view_long_dfs, view_metas)
    report_expected_but_missing(view_long_dfs)

    combined = pd.concat(view_long_dfs.values(), ignore_index=True)

    # Each view's rows already carry their own "group" from attach_group();
    # drop any rows where that per-view merge didn't find a match.
    n_before = len(combined)
    combined = combined.dropna(subset=["group"])
    if len(combined) < n_before:
        print(f"\n[info] dropped {n_before - len(combined)} rows with no group "
              f"assignment (sample missing from that layer's contrast groups file)")

    # Drop missing values -- MOFA handles NaN internally via the long format,
    # but there's no point carrying rows with no value at all.
    combined = combined.dropna(subset=["value"])

    combined = combined[["sample", "group", "feature", "view", "value"]]

    OUTPUT_LONG_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_LONG_DF_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {OUTPUT_LONG_DF_PATH}")
    print(combined.groupby(["view", "group"]).size())

    # Sanity check: since "sample" IDs are layer-specific, the SAME sample
    # string could in principle appear in two views but refer to different
    # physical material. If your sample-ID scheme is truly independent per
    # layer (e.g. separate numbering per omics layer), this is harmless --
    # MOFA just sees them as unrelated rows tagged with different "view"
    # values. But if IDs are meant to be cross-referenced (e.g. same
    # biological replicate profiled in two layers under a shared ID), make
    # sure that convention actually holds; a silent ID collision across
    # layers with different meanings would corrupt the model without
    # raising an error.
    trans_samples = set(view_long_dfs.get("transcriptomics", pd.DataFrame({"sample": []}))["sample"])
    prot_samples = set(view_long_dfs.get("proteomics", pd.DataFrame({"sample": []}))["sample"])
    overlapping_ids = trans_samples & prot_samples
    if overlapping_ids:
        print(f"\n[note] {len(overlapping_ids)} sample IDs appear in both "
              f"transcriptomics and proteomics data -- confirm this is intentional "
              f"(shared biological replicate ID) and not a coincidental string collision.")
    non_overlapping = trans_samples.symmetric_difference(prot_samples)
    if non_overlapping and (trans_samples and prot_samples):
        only_trans = trans_samples - prot_samples
        only_prot = prot_samples - trans_samples
        print(f"[note] sample IDs NOT shared between transcriptomics and proteomics:")
        if only_trans:
            print(f"    only in transcriptomics: {sorted(only_trans)}")
        if only_prot:
            print(f"    only in proteomics:      {sorted(only_prot)}")


if __name__ == "__main__":
    main()