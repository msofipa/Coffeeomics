from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

try:
    from mofapy2.run.entry_point import entry_point
except ImportError as e:
    msg = str(e)
    if "DLL load failed" in msg or "Application Control" in msg or "política" in msg.lower():
        raise ImportError(
            "mofapy2 IS installed -- this is a Windows Application Control "
            "policy (WDAC / Smart App Control) blocking a native DLL used by "
            "h5py, a mofapy2 dependency, from loading. Reinstalling mofapy2 or "
            "h5py will NOT fix this; it needs to be resolved at the OS/IT "
            "level (allowlist the blocked file, disable Smart App Control if "
            "this is a personal unmanaged device, or ask IT to approve it on "
            "a managed machine). See the original traceback above for exactly "
            "which file was blocked."
        ) from e
    raise ImportError(
        "mofapy2 (or one of its dependencies) failed to import. With the "
        "project venv active, run: pip install mofapy2"
    ) from e

# =========================================================================
# CONFIG 
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

# Output of build_mofa_input.py
LONG_INPUT_PATH = BASE_DIR / "integration" / "mofa2" / "mofa_input_long.csv"

# Where to save the trained model
OUTPUT_MODEL_PATH = BASE_DIR / "integration" / "mofa2" / "coffeeomics_mofa_model.hdf5"

# Where to save a first-look variance-explained summary
VARIANCE_EXPLAINED_CSV_PATH = BASE_DIR / "integration" / "mofa2" / "variance_explained.csv"

# --- Data options ---
SCALE_VIEWS = True  # put transcriptomics/proteomics/metabolomics on comparable scales

# --- Model options ---
N_FACTORS = 15          # generous starting point; prune later by variance explained
SPIKESLAB_WEIGHTS = True
ARD_WEIGHTS = True
# Required when using multiple groups (species, here) unless using MEFISTO --
# mofapy2 warns explicitly if this is left False with >1 group, since without
# it factor activity isn't modeled as potentially group-specific.
ARD_FACTORS = True

# --- Training options ---
N_ITER = 1000
CONVERGENCE_MODE = "medium"   # "fast" | "medium" | "slow"
DROP_R2 = None                # e.g. 0.01 to auto-drop near-zero-variance factors during training
GPU_MODE = False
SEED = 42

# =========================================================================
# Load assembled long-format input
# =========================================================================


def load_long_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Long-format MOFA input not found at {path}. "
            f"Run build_mofa_input.py first to generate it."
        )
    df = pd.read_csv(path)

    required_cols = {"sample", "group", "feature", "view", "value"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input file is missing required column(s): {missing_cols}. "
            f"mofapy2's set_data_df() expects columns "
            f"['sample','feature','view','group','value'] (order doesn't matter, "
            f"names do)."
        )

    print(f"Loaded {len(df)} rows from {path}")
    print(f"  views:  {sorted(df['view'].unique())}")
    print(f"  groups: {sorted(df['group'].unique())}")
    print("  samples per group:")
    print(df.groupby("group")["sample"].nunique().to_string())
    print("  features per view:")
    print(df.groupby("view")["feature"].nunique().to_string())

    return df


# =========================================================================
# Fit MOFA model
# =========================================================================


def fit_mofa(data: pd.DataFrame):
    ent = entry_point()

    ent.set_data_options(scale_views=SCALE_VIEWS)
    ent.set_data_df(data)

    ent.set_model_options(
        factors=N_FACTORS,
        spikeslab_weights=SPIKESLAB_WEIGHTS,
        ard_weights=ARD_WEIGHTS,
        ard_factors=ARD_FACTORS,
    )

    ent.set_train_options(
        iter=N_ITER,
        convergence_mode=CONVERGENCE_MODE,
        dropR2=DROP_R2,
        gpu_mode=GPU_MODE,
        seed=SEED,
    )

    ent.build()
    ent.run()

    return ent


# =========================================================================
# Post-training summary: variance explained per factor/view/group
# =========================================================================


def summarize_variance_explained(ent, output_csv: Path):
    """
    Extract variance explained (R2) per factor, per view, per group and
    save a tidy CSV -- the first thing worth checking post-fit to decide
    which of the N_FACTORS factors carry real signal vs. noise.

    mofapy2's internal return shape for calculate_variance_explained() has
    varied across versions, so this reshapes defensively: if the expected
    structure isn't there, it prints the raw object and pickles it instead
    of silently producing an empty/wrong table.
    """
    try:
        r2 = ent.model.calculate_variance_explained()
    except Exception as e:
        print(f"\n[warn] could not compute variance explained: {e}")
        return None

    try:
        records = []
        group_names = ent.data_opts.get("groups_names") if hasattr(ent.data_opts, "get") else None
        for group_idx, r2_group in enumerate(r2):
            group_name = group_names[group_idx] if group_names else f"group_{group_idx}"
            r2_group_df = pd.DataFrame(r2_group)
            for factor_idx, row in r2_group_df.iterrows():
                for view_name, value in row.items():
                    records.append({
                        "group": group_name,
                        "factor": f"Factor{factor_idx + 1}",
                        "view": view_name,
                        "r2": value,
                    })
        tidy = pd.DataFrame(records)
        if tidy.empty:
            raise ValueError("reshaped table is empty")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(output_csv, index=False)
        print(f"\nSaved variance-explained table to {output_csv}")

        total_per_factor = tidy.groupby("factor")["r2"].sum().sort_values(ascending=False)
        print("\nTotal variance explained per factor (summed across views/groups):")
        print(total_per_factor.to_string())

        return tidy

    except Exception as e:
        print(f"\n[warn] could not reshape variance-explained output into a tidy "
              f"table automatically ({e}).")
        print(f"[debug] raw type: {type(r2)}")
        raw_path = output_csv.with_suffix(".pkl")
        with open(raw_path, "wb") as f:
            pickle.dump(r2, f)
        print(f"Raw variance-explained object pickled to {raw_path} for manual "
              f"inspection -- structure may differ by mofapy2 version.")
        return None


# =========================================================================
# Main
# =========================================================================


def main():
    data = load_long_input(LONG_INPUT_PATH)

    print("\nFitting MOFA model ...")
    print(f"  factors={N_FACTORS}, scale_views={SCALE_VIEWS}, "
          f"iter={N_ITER}, convergence_mode={CONVERGENCE_MODE}, seed={SEED}")
    ent = fit_mofa(data)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ent.save(str(OUTPUT_MODEL_PATH), save_data=True)
    print(f"\nSaved trained model to {OUTPUT_MODEL_PATH}")

    summarize_variance_explained(ent, VARIANCE_EXPLAINED_CSV_PATH)


if __name__ == "__main__":
    main()