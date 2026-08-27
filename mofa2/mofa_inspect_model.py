"""
Diagnostic-only script: opens the ALREADY-TRAINED MOFA model file and
prints its internal HDF5 structure, so the variance-explained table in
fit_mofa_model.py can be reshaped correctly -- against ground truth,
instead of guessed.

Why this is needed: fit_mofa_model.py's summarize_variance_explained()
reshaped ent.model.calculate_variance_explained() assuming rows=factors,
columns=views. The resulting summary showed exactly 3 "factors" -- which
matches the VIEW count (metabolomics/proteomics/transcriptomics), not the
15 factors that were actually trained. That's a strong signal the
rows/columns axis assumption was backwards, and the table is very likely
reporting per-VIEW totals mislabeled as per-factor totals.

This script does not retrain anything -- it just opens the .hdf5 already
saved by fit_mofa_model.py and prints exactly what's inside, matching the
approach in the official template script (which reads
f["variance_explained"]["r2_per_factor"] / ["r2_total"] directly from the
file rather than from the in-memory model object).
"""

from pathlib import Path

import h5py

MODEL_PATH = Path(r"C:\Users\papan\Desktop\Coffeeomics\integration\mofa2\coffeeomics_mofa_model.hdf5")


def print_dataset_or_group(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  [dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"  [group]   {name}")


def dump_labels(f: h5py.File, key: str) -> None:
    if key not in f:
        return
    print(f"\n=== '{key}' ===")
    node = f[key]
    if isinstance(node, h5py.Dataset):
        arr = node[:]
        print(f"  shape={arr.shape}, dtype={arr.dtype}")
        print(f"  values: {arr}")
    else:
        for sub_key in node.keys():
            sub = node[sub_key]
            if isinstance(sub, h5py.Dataset):
                arr = sub[:]
                print(f"  {sub_key}: shape={arr.shape}, values={arr}")
            else:
                print(f"  [group] {sub_key}: sub-keys={list(sub.keys())}")


def dump_variance_explained(f: h5py.File) -> None:
    if "variance_explained" not in f:
        print("\n[!] No 'variance_explained' group found in this file.")
        return

    print("\n=== 'variance_explained' full contents ===")
    ve = f["variance_explained"]
    print(f"top-level keys: {list(ve.keys())}")

    for key in ve.keys():
        item = ve[key]
        print(f"\n[{key}]")
        if isinstance(item, h5py.Dataset):
            arr = item[:]
            print(f"  shape={arr.shape}, dtype={arr.dtype}")
            print(f"  values:\n{arr}")
        else:
            for sub_key in item.keys():
                sub_item = item[sub_key]
                if isinstance(sub_item, h5py.Dataset):
                    arr = sub_item[:]
                    print(f"  {sub_key}: shape={arr.shape}")
                    print(f"    values:\n{arr}")
                else:
                    print(f"  [group] {sub_key}: sub-keys={list(sub_item.keys())}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with h5py.File(MODEL_PATH, "r") as f:
        print("=== Top-level keys ===")
        print(list(f.keys()))

        print("\n=== Full recursive structure (all datasets/groups) ===")
        f.visititems(print_dataset_or_group)

        for key in ["views", "groups", "factors", "samples", "features"]:
            dump_labels(f, key)

        dump_variance_explained(f)


if __name__ == "__main__":
    main()