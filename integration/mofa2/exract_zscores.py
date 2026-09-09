"""
extract_factor_scores.py
-------------------------
Extracts the MOFA2 factor scores (Z matrix) from a trained model and saves
them as a long-format CSV: sample, group, factor, value.

Reads the HDF5 file directly via h5py (already a project dependency) rather
than requiring mofax or re-running mofapy2's entry_point -- this only reads
expectations already computed by fit_mofa_model.py, no retraining.

>>> Orientation caveat <<<
mofapy2 stores Z per group under /expectations/Z/<group_name>, but the
axis order (factors x samples vs samples x factors) is not something this
script assumes blindly -- given the axis-swap bug already found in the
variance-explained reshaping, orientation here is inferred by matching
each axis length against the known sample count for that group (from
/samples/<group_name>), and the inferred orientation is printed for you
to sanity check before trusting the output.

>>> Factor naming caveat <<<
get_factor_names() checks a couple of plausible HDF5 paths for stored
factor names/order, but this is unconfirmed against your actual file
structure (varies by mofapy2 version) -- if none match, it falls back to
the Factor{i+1} convention already used in fit_mofa_model.py's
summarize_variance_explained(), which is almost certainly what you want
anyway for consistency across outputs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

# =========================================================================
# CONFIG
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

MODEL_PATH = BASE_DIR / "integration" / "mofa2" / "coffeeomics_mofa_model.hdf5"
OUTPUT_CSV_PATH = BASE_DIR / "integration" / "mofa2" / "factor_scores_long.csv"


# =========================================================================
# Extraction
# =========================================================================


def get_factor_names(f: h5py.File, n_factors: int) -> list[str]:
    for candidate_path in ("factors", "expectations/AlphaZ/factors"):
        if candidate_path in f:
            try:
                names = [n.decode() if isinstance(n, bytes) else str(n)
                         for n in f[candidate_path][:]]
                if len(names) == n_factors:
                    return names
            except Exception:
                pass
    return [f"Factor{i + 1}" for i in range(n_factors)]


def extract_group_z(z_dataset: h5py.Dataset, sample_names: list[str],
                     factor_names: list[str], group_name: str) -> pd.DataFrame:
    """
    Orient a single group's Z dataset to (n_samples, n_factors) by matching
    axis lengths against the known sample count, then melt to long format.
    Raises loudly rather than guessing if neither axis matches.
    """
    z = z_dataset[:]
    n_samples = len(sample_names)
    n_factors = len(factor_names)

    if z.shape == (n_samples, n_factors):
        oriented = z
    elif z.shape == (n_factors, n_samples):
        print(f"  [{group_name}] Z stored as (factors, samples) -- transposing")
        oriented = z.T
    else:
        raise ValueError(
            f"[{group_name}] Z shape {z.shape} matches neither "
            f"(n_samples={n_samples}, n_factors={n_factors}) nor the "
            f"transpose -- cannot safely orient. Inspect the HDF5 structure "
            f"manually before trusting any output."
        )

    print(f"  [{group_name}] {oriented.shape[0]} samples x {oriented.shape[1]} factors")

    df = pd.DataFrame(oriented, index=sample_names, columns=factor_names)
    df.index.name = "sample"
    long_df = df.reset_index().melt(id_vars="sample", var_name="factor", value_name="value")
    long_df["group"] = group_name
    return long_df


def extract_all_factor_scores(model_path: Path) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    with h5py.File(model_path, "r") as f:
        if "expectations" not in f or "Z" not in f["expectations"]:
            raise KeyError(
                "Could not find /expectations/Z in the HDF5 file -- was this "
                "model saved with save_data=True / standard mofapy2 .save()? "
                f"Top-level keys found: {list(f.keys())}"
            )

        z_group = f["expectations"]["Z"]
        group_names = list(z_group.keys())
        print(f"Found {len(group_names)} group(s): {group_names}")

        all_long = []
        for group_name in group_names:
            sample_path = f"samples/{group_name}"
            if sample_path not in f:
                raise KeyError(
                    f"Could not find {sample_path} for sample names -- "
                    f"top-level 'samples' keys: {list(f.get('samples', {}).keys())}"
                )
            sample_names = [s.decode() if isinstance(s, bytes) else str(s)
                             for s in f[sample_path][:]]

            z_dataset = z_group[group_name]
            n_factors_candidates = [dim for dim in z_dataset.shape if dim != len(sample_names)]
            n_factors = n_factors_candidates[0] if n_factors_candidates else z_dataset.shape[0]
            factor_names = get_factor_names(f, n_factors)

            long_df = extract_group_z(z_dataset, sample_names, factor_names, group_name)
            all_long.append(long_df)

    combined = pd.concat(all_long, ignore_index=True)
    combined = combined[["sample", "group", "factor", "value"]]
    return combined


# =========================================================================
# Main
# =========================================================================


def main():
    print(f"Reading trained model from {MODEL_PATH} ...")
    long_df = extract_all_factor_scores(MODEL_PATH)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nWrote {len(long_df)} rows to {OUTPUT_CSV_PATH}")
    print(long_df.groupby('group')['sample'].nunique())


if __name__ == "__main__":
    main()