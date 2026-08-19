"""
shapiro_residuals_analysis.py
──────────────────────────────
Tests whether metabolite abundance RESIDUALS are normally distributed, via
the Shapiro-Wilk test — the diagnostic used to justify choosing the
Mann-Whitney U test over one-way ANOVA for condition-wise comparisons.

Logic
─────
For each species x temperature x CO2 x water-stress group (the same
"contrast group" structure used throughout the rest of the pipeline), the
group mean is computed per metabolite, and each sample's RESIDUAL is its
raw abundance minus that group's mean (see the "what are residuals"
explanation from our conversation — ANOVA's normality assumption is on
these residuals, not on the raw abundance values).

Residuals are pooled ACROSS ALL groups for a given metabolite (this is the
standard way to check the normality assumption for a multi-group ANOVA:
one pooled residual distribution per response variable, not one test per
group), then the Shapiro-Wilk test is run once per metabolite on that
pooled residual vector.

Groups with fewer than 2 samples are excluded from residual computation —
a single-sample group has a trivial (zero) residual that would bias the
pooled distribution toward 0 without carrying any real information.

Two diagnostic plots for heteroscedasticity are also produced (see
plot_heteroscedasticity()):
  1. Residuals vs. fitted values — the standard diagnostic scatter: if
     variance is homogeneous (homoscedastic), residual spread should be
     roughly constant across the range of fitted (group mean) values; a
     "fan" shape (spread increasing with fitted value) indicates
     heteroscedasticity.
  2. Group standard deviation vs. group mean, per metabolite x group — a
     direct look at the mean-variance relationship. An increasing trend
     (variance scaling with the mean) is the signature of the log-normal-
     like behaviour typical of abundance/intensity data.

Output
──────
    shapiro_residuals_summary.csv     — one row per metabolite: n residuals,
                                         W statistic, p-value, normal (bool)
    shapiro_residuals_summary.xlsx    — same table, with non-normal rows
                                         (p < 0.05) highlighted
    residuals_vs_fitted.png           — heteroscedasticity diagnostic plot 1
    group_mean_vs_std.png             — heteroscedasticity diagnostic plot 2
    Console summary                   — counts/percentage of metabolites
                                         failing normality, ready to quote
                                         directly in the methodology text
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro

import matplotlib
matplotlib.use("Agg")   # headless-safe backend; also fine when run interactively
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — same file layout / condition structure as mann_whitney.ipynb
# ─────────────────────────────────────────────────────────────────────────

SAMPLE_IDS_FILE = r"C:\Users\papan\Desktop\Coffeeomics\sample_information.csv"
METABOLOMICS_FILE = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\data\metabolomics_counts.csv"
METABOLOMICS_RESULTS_FOLDER = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\shapiro_wilk_results"

P_VALUE_THRESHOLD = 0.05   # normality rejected below this
MIN_GROUP_SIZE    = 2      # groups smaller than this contribute no residual

CONDITIONS = {
    "species": ["C. arabica", "C. canephora"],
    "temperature": ("25/20°C", "37/28°C", "42/30°C", "Rec 14 - 25/20°C"),
    "co2": ("380ppm", "700ppm"),
    "drought": ("without", "medium", "severe"),
}

# ─────────────────────────────────────────────────────────────────────────
# LOADING  (mirrors the corrected mann_whitney.ipynb cells)
# ─────────────────────────────────────────────────────────────────────────

def load_samples():
    samples = pd.read_csv(SAMPLE_IDS_FILE, index_col=0)
    samples = samples.rename(columns={
        "Species": "species",
        "Temperature": "temperature",
        "CO2_ppm": "co2",
        "Water_Stress": "drought",
    })
    samples["co2"] = samples["co2"].astype(str)

    # Keep only samples actually profiled for metabolomics
    samples = samples[samples["Metabolomics"]]
    return samples


def load_metabolomics(samples):
    # metabolomics_counts.csv has a 5-row header (Species, Temperature, CO2,
    # Water_Stress, Sample_ID) — must be read as a MultiIndex header, then
    # flattened to the Sample_ID level only.
    metabolomics = pd.read_csv(METABOLOMICS_FILE, index_col=0, header=[0, 1, 2, 3, 4])
    metabolomics.columns = metabolomics.columns.get_level_values(-1)

    # Reconcile against the actual data columns (metadata flag and data can disagree)
    only_in_metadata = set(samples.index) - set(metabolomics.columns)
    only_in_data      = set(metabolomics.columns) - set(samples.index)
    if only_in_metadata:
        print(f"Warning: {len(only_in_metadata)} sample(s) flagged Metabolomics=True in "
              f"sample_information.csv but missing from metabolomics_counts.csv: "
              f"{sorted(only_in_metadata)}")
    if only_in_data:
        print(f"Warning: {len(only_in_data)} sample(s) present in metabolomics_counts.csv "
              f"but not flagged Metabolomics=True (or absent) in sample_information.csv: "
              f"{sorted(only_in_data)}")

    return metabolomics


# ─────────────────────────────────────────────────────────────────────────
# RESIDUAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────

def compute_pooled_residuals(samples, metabolomics):
    """
    Returns:
      residuals   — DataFrame (metabolites x samples): each sample's value
                    minus the mean of its own species x temperature x CO2 x
                    drought group, restricted to groups with >= MIN_GROUP_SIZE
                    members (smaller groups' samples are dropped entirely).
      fitted      — DataFrame, same shape/index/columns as `residuals`: the
                    group mean ("fitted value") each residual was computed
                    against — needed for the residuals-vs-fitted plot.
      group_stats — DataFrame with one row per (metabolite, group): group
                    mean and group standard deviation — needed for the
                    mean-vs-std plot.
    """
    residuals = pd.DataFrame(index=metabolomics.index)
    fitted    = pd.DataFrame(index=metabolomics.index)
    group_stats_rows = []
    skipped_groups = []

    for species in CONDITIONS["species"]:
        for temperature in CONDITIONS["temperature"]:
            for co2 in CONDITIONS["co2"]:
                for drought in CONDITIONS["drought"]:
                    group_ids = samples[
                        (samples["species"] == species) &
                        (samples["temperature"] == temperature) &
                        (samples["co2"] == co2) &
                        (samples["drought"] == drought)
                    ].index.tolist()
                    group_ids = [s for s in group_ids if s in metabolomics.columns]

                    if len(group_ids) < MIN_GROUP_SIZE:
                        if group_ids:
                            skipped_groups.append(
                                f"{species} | {temperature} | {co2} | {drought} (n={len(group_ids)})"
                            )
                        continue

                    group_vals = metabolomics[group_ids]
                    group_mean = group_vals.mean(axis=1)
                    group_std  = group_vals.std(axis=1)
                    group_resid = group_vals.sub(group_mean, axis=0)
                    group_fitted = pd.DataFrame(
                        {s: group_mean for s in group_ids}, index=metabolomics.index
                    )

                    residuals = residuals.join(group_resid, how="left")
                    fitted    = fitted.join(group_fitted, how="left")

                    group_stats_rows.append(pd.DataFrame({
                        "metabolite": metabolomics.index,
                        "group": f"{species} | {temperature} | {co2} | {drought}",
                        "n": len(group_ids),
                        "mean": group_mean.values,
                        "std": group_std.values,
                    }))

    if skipped_groups:
        print(f"\nSkipped {len(skipped_groups)} group(s) with fewer than "
              f"{MIN_GROUP_SIZE} sample(s) (no residual computed):")
        for g in skipped_groups:
            print(f"  {g}")

    group_stats = pd.concat(group_stats_rows, ignore_index=True)
    return residuals, fitted, group_stats


# ─────────────────────────────────────────────────────────────────────────
# SHAPIRO-WILK TEST  (one test per metabolite, on its pooled residuals)
# ─────────────────────────────────────────────────────────────────────────

def run_shapiro_per_metabolite(residuals):
    rows = []
    for metabolite, row in residuals.iterrows():
        vals = row.dropna().values
        n = len(vals)

        if n < 3:
            # Shapiro-Wilk requires at least 3 observations
            rows.append({"metabolite": metabolite, "n_residuals": n,
                         "W": np.nan, "p_value": np.nan, "normal": np.nan})
            continue

        w_stat, p_val = shapiro(vals)
        rows.append({
            "metabolite": metabolite,
            "n_residuals": n,
            "W": w_stat,
            "p_value": p_val,
            "normal": p_val >= P_VALUE_THRESHOLD,
        })

    return pd.DataFrame(rows).set_index("metabolite")


# ─────────────────────────────────────────────────────────────────────────
# HETEROSCEDASTICITY PLOTS
# ─────────────────────────────────────────────────────────────────────────

def plot_heteroscedasticity(residuals, fitted, group_stats, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: residuals vs. fitted values (pooled across all metabolites) ──
    fitted_flat    = fitted.values.flatten()
    residuals_flat = residuals.values.flatten()
    mask = ~np.isnan(fitted_flat) & ~np.isnan(residuals_flat)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted_flat[mask], residuals_flat[mask], s=8, alpha=0.3, edgecolor="none")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted value (group mean abundance)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs. Fitted Values\n(pooled across all metabolites and groups)")
    fig.tight_layout()
    path1 = out_dir / "residuals_vs_fitted.png"
    fig.savefig(path1, dpi=300)
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Plot 2: group standard deviation vs. group mean, log-log ──────────
    valid = group_stats[(group_stats["n"] >= MIN_GROUP_SIZE) & (group_stats["mean"] > 0) & (group_stats["std"] > 0)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["mean"], valid["std"], s=8, alpha=0.3, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Group mean abundance (log scale)")
    ax.set_ylabel("Group standard deviation (log scale)")
    ax.set_title("Group Mean vs. Group Standard Deviation\n(one point per metabolite x condition group)")
    fig.tight_layout()
    path2 = out_dir / "group_mean_vs_std.png"
    fig.savefig(path2, dpi=300)
    plt.close(fig)
    print(f"Saved: {path2}")


# ─────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────

def export_results(summary_df):
    out_dir = Path(METABOLOMICS_RESULTS_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "shapiro_residuals_summary.csv"
    summary_df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    xlsx_path = out_dir / "shapiro_residuals_summary.xlsx"
    try:
        styled = summary_df.style.map(
            lambda v: "background-color: #e06666" if v is False else "",
            subset=["normal"]
        )
        styled.to_excel(xlsx_path, engine="openpyxl")
        print(f"Saved: {xlsx_path}")
    except ImportError:
        # .style requires jinja2; fall back to a plain (unstyled) export
        summary_df.to_excel(xlsx_path, engine="openpyxl")
        print(f"Saved: {xlsx_path}  (unstyled — install jinja2 for highlighting)")


# ─────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTOUT — numbers ready to quote in the methodology text
# ─────────────────────────────────────────────────────────────────────────

def print_summary(summary_df):
    tested = summary_df.dropna(subset=["p_value"])
    n_tested = len(tested)
    n_nonnormal = int((tested["p_value"] < P_VALUE_THRESHOLD).sum())
    pct_nonnormal = 100 * n_nonnormal / n_tested if n_tested else float("nan")

    print(f"\n{'=' * 60}")
    print("Shapiro-Wilk normality summary (pooled residuals)")
    print(f"{'=' * 60}")
    print(f"Metabolites tested:                {n_tested}")
    print(f"Metabolites excluded (n < 3):       {len(summary_df) - n_tested}")
    print(f"Non-normal residuals (p < {P_VALUE_THRESHOLD}):  {n_nonnormal} "
          f"({pct_nonnormal:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    samples = load_samples()
    metabolomics = load_metabolomics(samples)

    residuals, fitted, group_stats = compute_pooled_residuals(samples, metabolomics)
    summary_df = run_shapiro_per_metabolite(residuals)

    export_results(summary_df)
    plot_heteroscedasticity(residuals, fitted, group_stats, METABOLOMICS_RESULTS_FOLDER)
    print_summary(summary_df)


if __name__ == "__main__":
    main()