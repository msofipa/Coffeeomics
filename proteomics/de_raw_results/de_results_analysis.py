import os
import pandas as pd

# ─── PATHS ────────────────────────────────────────────────────────────────────
MARGINAL_DIR = r"C:\Users\papan\Desktop\Coffeeomics\proteomics\de_raw_results\marginal"

# Treatment (table row) → source file, in the order they appear in the table.
FACTOR_FILES = {
    "CO$_2$":      "co2_de_raw_results.xlsx",
    "Temperature": "temperature_de_raw_results.xlsx",
    "Water":       "condition_de_raw_results.xlsx",
}

# Species (Excel column prefix) → table row label, in table order.
SPECIES_LABELS = {
    "C. arabica":   "Icatu",
    "C. canephora": "CL153",
}

OUTPUT_TEX = os.path.join(MARGINAL_DIR, "sdeg_summary_table.tex")
OUTPUT_CSV = os.path.join(MARGINAL_DIR, "sdeg_summary_table.csv")


# ─── COMPUTE SUMMARY ────────────────────────────────────────────────────────────

def compute_species_summary(factor_file: str, species: str) -> tuple[int, int]:
    """Return (unique_sdegs, total_instances) for one species within one factor file."""
    path = os.path.join(MARGINAL_DIR, factor_file)
    if not os.path.exists(path):
        print(f"WARNING: missing file, skipping: {path}")
        return 0, 0

    df = pd.read_excel(path, sheet_name="log2FoldChange")
    df = df.set_index("protein")

    species_cols = [c for c in df.columns if c.startswith(f"{species} | ")]
    if not species_cols:
        print(f"WARNING: no columns found for species '{species}' in {factor_file}")
        return 0, 0

    sub = df[species_cols]
    non_null = sub.notna()

    unique_sdegs = int(non_null.any(axis=1).sum())
    total_instances = int(non_null.values.sum())

    return unique_sdegs, total_instances


def build_summary() -> dict[tuple[str, str], tuple[int, int]]:
    """{(species_label, treatment_label): (unique_sdegs, total_instances)}"""
    summary = {}
    for treatment, factor_file in FACTOR_FILES.items():
        for species, species_label in SPECIES_LABELS.items():
            unique_sdegs, total_instances = compute_species_summary(factor_file, species)
            summary[(species_label, treatment)] = (unique_sdegs, total_instances)
            print(f"{species_label:8s} | {treatment:12s} | unique={unique_sdegs:5d} | total={total_instances:5d}")
    return summary


# ─── OUTPUT ───────────────────────────────────────────────────────────────────

def save_csv(summary: dict[tuple[str, str], tuple[int, int]]) -> None:
    rows = []
    for (species_label, treatment), (unique_sdegs, total_instances) in summary.items():
        rows.append({
            "species": species_label,
            "treatment": treatment,
            "unique_sdegs": unique_sdegs,
            "total_instances": total_instances,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")


def save_latex(summary: dict[tuple[str, str], tuple[int, int]]) -> None:
    def cell(species_label, treatment):
        return summary.get((species_label, treatment), (0, 0))

    icatu_co2   = cell("Icatu", "CO$_2$")
    icatu_temp  = cell("Icatu", "Temperature")
    icatu_water = cell("Icatu", "Water")
    cl153_co2   = cell("CL153", "CO$_2$")
    cl153_temp  = cell("CL153", "Temperature")
    cl153_water = cell("CL153", "Water")

    tex = r"""\begin{table}
\centering
\caption{Summary of significant DEGs by omics layer and species for aggregated data}
\label{tab:deg_prot_summary}
\begin{tabularx}{\textwidth}{@{}llcXX@{}}
\toprule
Layer & Species & Treatment & Unique SDEGs & Total Instances of SDEGs \\
\midrule
\multirow{6}{*}{Proteomics}
    & \multirow{3}{*}{\textbf{Icatu}$^{*}$}
        & CO$_2$       & %d & %d \\
    &   & Temperature & %d & %d \\
    &   & Water       & %d & %d \\
    \cmidrule(l){2-5}
    & \multirow{3}{*}{\textbf{CL153}$^{**}$}
        & CO$_2$       & %d & %d \\
    &   & Temperature & %d & %d \\
    &   & Water       & %d & %d \\
\bottomrule
\end{tabularx}
\vspace{0.5em}
\footnotesize
$^{*}$ \textit{C. arabica}\\ Total Instances of SDEGs are all instances for which we have a number (all entries of the matrix that are non nan) and all genes (rows) for wich at least one entrie is non nan
$^{**}$ \textit{C. canephora}
\end{table}
""" % (
        icatu_co2[0], icatu_co2[1],
        icatu_temp[0], icatu_temp[1],
        icatu_water[0], icatu_water[1],
        cl153_co2[0], cl153_co2[1],
        cl153_temp[0], cl153_temp[1],
        cl153_water[0], cl153_water[1],
    )

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Saved → {OUTPUT_TEX}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = build_summary()
    save_csv(summary)
    save_latex(summary)

