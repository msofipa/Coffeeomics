import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

TRANSCRIPTOMICS_PATH = BASE_DIR / "transcriptomics" / "normalizations" / "trans_pydeseq2_normalized_counts.csv"
PROTEOMICS_PATH = BASE_DIR / "proteomics" / "data" / "prot_normalized_pydeseq2_manual_50_above_10.xlsx"
METABOLOMICS_PATH = BASE_DIR / "metabolomics" / "data" / "metabolomics_counts.csv"

# --------------------------------------------------------------------------
# COUNTS-MATRIX LOADING / 0s-1s-NaNs SUMMARY
# --------------------------------------------------------------------------

def load_counts_matrix(path, header_rows=1):
    """
    Loads a counts/abundance matrix from .csv or .xlsx, trying utf-8 first
    and falling back to cp1252 for CSVs (per project convention for files
    with degree symbols). Assumes the first column is the feature index
    (index_col=0).

    header_rows: number of header rows. Metabolomics uses a 5-level
    MultiIndex header (Species, Temperature, CO2, Water Stress, Sample ID);
    other layers use a single header row.
    """
    path = Path(path)
    header = list(range(header_rows)) if header_rows > 1 else 0

    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=0, header=header)

    try:
        return pd.read_csv(path, index_col=0, header=header, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, index_col=0, header=header, encoding="cp1252")


def count_zero_one_nan(df, label=None):
    """
    Reports the number of 0s, 1s, and NaNs across ALL entries of a
    counts/abundance matrix, each shown as a fraction of the total
    number of entries, plus shape for context.
    """
    total_entries = df.size

    nan_count = df.isna().sum().sum()
    zero_count = (df == 0.0).sum().sum()
    one_count = (df == 1.0).sum().sum()

    if label:
        print(f"\n=== {label} ===")
    print(f"Shape: {df.shape}  ({df.shape[0]} features x {df.shape[1]} samples)")
    print(f"Total entries: {total_entries}")
    print(f"NaNs: {nan_count}/{total_entries} ({nan_count / total_entries:.2%})")
    print(f"Zeros: {zero_count}/{total_entries} ({zero_count / total_entries:.2%})")
    print(f"Ones: {one_count}/{total_entries} ({one_count / total_entries:.2%})")


def summarize_all_counts_matrices():
    """
    Loads the transcriptomics, proteomics, and metabolomics counts matrices
    and reports 0/1/NaN counts for each.
    """
    layers = [
        ("Transcriptomics", TRANSCRIPTOMICS_PATH, 1),
        ("Proteomics", PROTEOMICS_PATH, 1),
        ("Metabolomics", METABOLOMICS_PATH, 5),
    ]

    for label, path, header_rows in layers:
        df = load_counts_matrix(path, header_rows=header_rows)
        count_zero_one_nan(df, label=label)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

if __name__ == "__main__":
    summarize_all_counts_matrices()