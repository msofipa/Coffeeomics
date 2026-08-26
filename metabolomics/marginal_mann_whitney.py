"""
Marginal (pooled) Mann-Whitney U pipeline — Coffeeomics metabolomics.

Companion to mann_whitney.ipynb's fully-factorial comparisons (every
temperature/co2/drought level held fixed except the one pair being
contrasted). Here, for a given factor of interest (temperature, co2, or
drought), samples are pooled across all levels of the two remaining
(non-focal) factors, WITHIN each species — yielding one aggregated sample
group per (species, factor level). Pairwise Mann-Whitney U tests are then
run between these pooled groups, same species, same factor, using the
identical scipy workflow, significance threshold, and highlighted-Excel
export style as mann_whitney.ipynb.

Sample loading, metabolomics-matrix loading, the metadata/data reconciliation
warnings, run_mannwhitneyu(), and the p-value highlighting in the export are
carried over from the notebook essentially unchanged — only the grouping
logic (fully-factorial → pooled-by-factor) and the output layout
(per-comparison sheets → one workbook per factor, one sheet per species)
are new.

Recovery ("Rec 14 - 25/20°C") is a level of temperature. Recovery-condition
samples are excluded prior to pooling specifically for the
"temperature-collapsed" analyses — i.e. whenever temperature is one of the
two non-focal factors being pooled away (factor of interest = co2 or
drought). When temperature itself is the factor of interest, the recovery
level is a real level being contrasted and is never excluded. This mirrors
the convention already confirmed for transcriptomics/proteomics — verify
RECOVERY_LEVEL's exact spelling against your data if it ever changes.
"""

import json
import os

import pandas as pd
from scipy.stats import mannwhitneyu

# ─── PATHS ────────────────────────────────────────────────────────────────────
SAMPLE_IDS_FILE = r"C:\Users\papan\Desktop\Coffeeomics\sample_information.csv"
METABOLOMICS_FILE = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\data\metabolomics_counts.csv"

# All marginal-analysis output (per-factor groupings JSON + per-factor
# highlighted Excel results) is written into this one folder.
MARGINAL_DIR = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\mann_whitney_results\marginal"

P_VALUE_THRESHOLD = 0.05

# ─── METADATA STRUCTURE ───────────────────────────────────────────────────────
SPECIES_LIST = ["C. arabica", "C. canephora"]

# Factors eligible to be the "factor of interest" for a marginal analysis.
# species is never a factor of interest — pooling always happens within species.
MARGINAL_FACTORS = ["temperature", "co2", "drought"]

# Exact label used for the post-stress recovery level within temperature.
# *** VERIFY against your actual data — adjust to match exactly. ***
RECOVERY_LEVEL = "Rec 14 - 25/20°C"
TEMPERATURE_COL = "temperature"

# Factors of interest for which recovery-condition samples are excluded prior
# to pooling — i.e. whenever temperature (which contains the recovery level)
# is one of the two non-focal factors being pooled/collapsed away. This is
# the "temperature-collapsed analysis": factor of interest ≠ temperature.
EXCLUDE_RECOVERY_FOR_FACTORS = {"co2", "drought"}


# ─── LOAD SAMPLES + METABOLOMICS DATA (unchanged from mann_whitney.ipynb) ──────

def load_samples() -> pd.DataFrame:
    samples = pd.read_csv(SAMPLE_IDS_FILE, index_col=0)
    samples = samples.rename(columns={
        "Species": "species",
        "Temperature": "temperature",
        "CO2_ppm": "co2",
        "Water_Stress": "drought",
    })
    samples["co2"] = samples["co2"].astype(str)

    # sample_information.csv lists samples across ALL omics layers; keep only
    # those actually profiled for metabolomics, or downstream lookups (e.g. '7A',
    # which has Metabolomics=False) raise a KeyError against metabolomics_counts.csv
    samples = samples[samples["Metabolomics"]]
    return samples


def load_metabolomics() -> pd.DataFrame:
    # metabolomics_counts.csv has a 5-row header (Species, Temperature, CO2, Water_Stress,
    # Sample_ID) -- must be read as a MultiIndex header, then flattened to the Sample_ID
    # level only.
    metabolomics = pd.read_csv(METABOLOMICS_FILE, index_col=0, header=[0, 1, 2, 3, 4])
    metabolomics.columns = metabolomics.columns.get_level_values(-1)  # keep only Sample_ID
    return metabolomics


def reconcile(samples: pd.DataFrame, metabolomics: pd.DataFrame) -> pd.DataFrame:
    # sample_information.csv's Metabolomics flag and metabolomics_counts.csv's columns
    # can disagree (e.g. a sample flagged True but later dropped in QC, or added to the
    # data without the metadata being updated).
    only_in_metadata = set(samples.index) - set(metabolomics.columns)
    only_in_data = set(metabolomics.columns) - set(samples.index)
    if only_in_metadata:
        print(f"Warning: {len(only_in_metadata)} sample(s) flagged Metabolomics=True in "
              f"sample_information.csv but missing from metabolomics_counts.csv: "
              f"{sorted(only_in_metadata)}")
    if only_in_data:
        print(f"Warning: {len(only_in_data)} sample(s) present in metabolomics_counts.csv "
              f"but not flagged Metabolomics=True (or absent) in sample_information.csv: "
              f"{sorted(only_in_data)}")
    return samples[samples.index.isin(metabolomics.columns)]


# ─── BUILD POOLED GROUPS ────────────────────────────────────────────────────────

def build_pooled_groups(
    samples: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """
    For every (species, factor_of_interest) pair, pool sample IDs across all
    levels of the two remaining (non-focal) factors, and group them by the
    factor-of-interest's level.

    Returns: { (species, factor): { level: [sample_id, ...], ... }, ... }
    """
    pooled: dict[tuple[str, str], dict[str, list[str]]] = {}

    for species in SPECIES_LIST:
        species_info = samples[samples["species"] == species]

        for factor in MARGINAL_FACTORS:
            subset = species_info
            if factor in EXCLUDE_RECOVERY_FOR_FACTORS:
                before = len(subset)
                subset = subset[subset[TEMPERATURE_COL] != RECOVERY_LEVEL]
                dropped = before - len(subset)
                if dropped:
                    print(f"  [{species} | {factor}] excluded {dropped} recovery-condition sample(s) prior to pooling")

            levels: dict[str, list[str]] = {}
            for level, grp in subset.groupby(factor):
                sample_ids = grp.index.values.tolist()
                if sample_ids:
                    levels[level] = sample_ids

            if len(levels) >= 2:
                pooled[(species, factor)] = levels
            else:
                print(f"  [{species} | {factor}] skipped — fewer than 2 pooled levels available")

    return pooled


def save_pooled_groups_json(
    pooled_groups: dict[tuple[str, str], dict[str, list[str]]],
) -> None:
    """One groups JSON per factor, written to MARGINAL_DIR."""
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
        print(f"Saved pooled groupings for {factor} ({len(flat)} groups) → {path}")


# ─── MANN-WHITNEY U TEST (unchanged from mann_whitney.ipynb) ──────────────────

def run_mannwhitneyu(metabolomics: pd.DataFrame, a_sample_ids: list[str], b_sample_ids: list[str]) -> pd.DataFrame:
    a = metabolomics[a_sample_ids]
    b = metabolomics[b_sample_ids]
    result_df = a.apply(
        lambda row: mannwhitneyu(
            row.dropna(),
            b.loc[row.name].dropna(),
            alternative="two-sided"
        ),
        axis=1,
        result_type="expand"
    )
    result_df.columns = ["u_statistic", "p_value"]
    return result_df


# ─── RUN MARGINAL COMPARISONS ───────────────────────────────────────────────────

def run_marginal_mannwhitneyu(
    metabolomics: pd.DataFrame,
    pooled_groups: dict[tuple[str, str], dict[str, list[str]]],
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run every pooled pairwise contrast and organize results as:
      { factor: { species: sheet_df } }
    where sheet_df has MultiIndex columns keyed by "{level_a} vs {level_b}",
    each holding [u_statistic, p_value] — mirroring the notebook's
    concat-and-key pattern.
    """
    from itertools import combinations

    results: dict[str, dict[str, pd.DataFrame]] = {factor: {} for factor in MARGINAL_FACTORS}

    for (species, factor), levels in pooled_groups.items():
        level_names = sorted(levels.keys())
        comp_dfs = []
        comp_names = []

        for level_a, level_b in combinations(level_names, 2):
            a_ids = levels[level_a]
            b_ids = levels[level_b]
            print(f"\nRunning: {species} | {factor}: {level_a} vs {level_b}")
            result_df = run_mannwhitneyu(metabolomics, a_ids, b_ids)
            comp_dfs.append(result_df)
            comp_names.append(f"{level_a} vs {level_b}")

        if comp_dfs:
            sheet_df = pd.concat(comp_dfs, axis=1, keys=comp_names)
            results[factor][species] = sheet_df

    return results


# ─── EXPORT (highlighting logic unchanged from mann_whitney.ipynb) ─────────────

def export_mannwhitneyu_marginal(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    os.makedirs(MARGINAL_DIR, exist_ok=True)

    for factor, sheets_by_species in results.items():
        if not sheets_by_species:
            continue

        output_path = os.path.join(MARGINAL_DIR, f"{factor}_mannwhitneyu_marginal.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for species, df in sheets_by_species.items():
                # Only highlight p_value columns -- df.style.map() applies to every cell,
                # so without this subset it was also (meaninglessly) highlighting any
                # u_statistic that happened to fall between 0 and P_VALUE_THRESHOLD.
                p_value_cols = [c for c in df.columns if c[-1] == "p_value"]
                styled = df.style.map(
                    lambda v: "background-color: #32a852"
                    if P_VALUE_THRESHOLD >= v > 0 else "",
                    subset=p_value_cols
                )
                styled.to_excel(writer, sheet_name=species, merge_cells=True)

        print(f"Saved → {output_path}")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def pipeline(groups_only: bool = False) -> None:
    samples = load_samples()
    metabolomics = load_metabolomics()
    samples = reconcile(samples, metabolomics)

    pooled_groups = build_pooled_groups(samples)
    save_pooled_groups_json(pooled_groups)

    if groups_only:
        print(f"\ngroups-only mode: skipping Mann-Whitney tests. Inspect the *_groups.json files in {MARGINAL_DIR}, then re-run without --groups-only.")
        return

    results = run_marginal_mannwhitneyu(metabolomics, pooled_groups)
    export_mannwhitneyu_marginal(results)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pipeline(groups_only="--groups-only" in sys.argv)