"""
Build a contrast_groups_<layer>.json file from sample_information.csv.

For a chosen omics layer (Transcriptomics, Proteomic, or Metabolomics),
this script:
  1. Loads the sample sheet.
  2. Filters to rows where the layer's column is True.
  3. Groups the remaining Sample_IDs by their condition
     (Species | Temperature | CO2_ppm | Water_Stress).
  4. Writes the result as a JSON file with the same structure as
     contrast_groups_proteomics.json:

        {
          "C. arabica | 25/20°C | 380ppm | medium": ["20B", "20C", "8A"],
          ...
        }

Usage (edit the CONFIG block below, or pass args on the command line):
    python build_contrast_groups.py --layer Transcriptomics
    python build_contrast_groups.py --layer Proteomic --csv "C:\\path\\to\\sample_information.csv"
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ----------------------------- CONFIG --------------------------------------
# Default paths (Windows). Override via command-line args if you prefer.
DEFAULT_CSV_PATH = r"C:\Users\papan\Desktop\Coffeeomics\sample_information.csv"
DEFAULT_OUTPUT_DIR = r"C:\Users\papan\Desktop\Coffeeomics"

# Which layer to process when running the script directly (e.g. from an IDE's
# "Run" button or a Jupyter notebook, with no command-line arguments).
# Set to one of: "Transcriptomics", "Proteomic", "Metabolomics"
LAYER = "Transcriptomics"

# Valid layer column names as they appear in the CSV header.
VALID_LAYERS = ["Transcriptomics", "Proteomic", "Metabolomics"]

# Columns that define a "condition" (i.e. a contrast group), in display order.
CONDITION_COLUMNS = ["Species", "Temperature", "CO2_ppm", "Water_Stress"]

SAMPLE_ID_COLUMN = "Sample_ID"
# -----------------------------------------------------------------------------


def load_sample_sheet(csv_path: Path) -> pd.DataFrame:
    """Load the sample information CSV, trying UTF-8 first (this file's actual
    encoding) and falling back to cp1252 for older Windows-exported files that
    use that encoding instead (e.g. degree symbols saved as 0xB0)."""
    log.info("Loading sample sheet from %s", csv_path)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        log.info("UTF-8 decode failed, falling back to cp1252")
        df = pd.read_csv(csv_path, encoding="cp1252", encoding_errors="replace")

    # Normalize boolean-like columns in case they were read as strings
    # (e.g. "True"/"False"/"TRUE"/"FALSE"/1/0).
    for layer in VALID_LAYERS:
        if layer not in df.columns:
            raise ValueError(
                f"Expected column '{layer}' not found in CSV. "
                f"Found columns: {list(df.columns)}"
            )
        if df[layer].dtype != bool:
            df[layer] = (
                df[layer]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "1": True, "false": False, "0": False})
            )

    missing_cols = [c for c in CONDITION_COLUMNS + [SAMPLE_ID_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected column(s): {missing_cols}")

    return df


def build_contrast_groups(df: pd.DataFrame, layer: str) -> dict:
    """Filter to samples included in `layer`, then group Sample_IDs by condition."""
    if layer not in VALID_LAYERS:
        raise ValueError(f"layer must be one of {VALID_LAYERS}, got '{layer}'")

    filtered = df[df[layer] == True].copy()  # noqa: E712
    log.info("Layer '%s': %d / %d samples included", layer, len(filtered), len(df))

    if filtered.empty:
        log.warning("No samples found for layer '%s' — check the CSV values.", layer)
        return {}

    # Build the condition key exactly as in contrast_groups_proteomics.json:
    # "Species | Temperature | CO2_ppm | Water_Stress"
    filtered["__condition_key__"] = filtered[CONDITION_COLUMNS].astype(str).agg(" | ".join, axis=1)

    contrast_groups: dict[str, list[str]] = {}
    # Preserve first-appearance order of conditions and of sample IDs within them.
    for condition_key, group_df in filtered.groupby("__condition_key__", sort=False):
        sample_ids = group_df[SAMPLE_ID_COLUMN].astype(str).tolist()
        contrast_groups[condition_key] = sample_ids

    return contrast_groups


def main():
    parser = argparse.ArgumentParser(description="Build contrast_groups_<layer>.json from sample_information.csv")
    parser.add_argument(
        "--layer",
        choices=VALID_LAYERS,
        default=LAYER,
        help=f"Omics layer to filter on (default: the LAYER variable in CONFIG, currently '{LAYER}')",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to sample_information.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Directory to write the output JSON to")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_sample_sheet(csv_path)
    contrast_groups = build_contrast_groups(df, args.layer)

    out_path = outdir / f"contrast_groups_{args.layer.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contrast_groups, f, indent=2, ensure_ascii=False)

    log.info("Wrote %d condition groups to %s", len(contrast_groups), out_path)


if __name__ == "__main__":
    main()