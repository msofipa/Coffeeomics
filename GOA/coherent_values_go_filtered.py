#!/usr/bin/env python3
import os
import csv
import subprocess
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR / "proteomics"  # change to "proteomics" if needed

INPUT_TYPE = "proteomics"   # "proteomics" | "transcriptomics"

INPUT_DATA       = BASE_DIR / "inputs" / "comparison_gene_lists"
OUTPUT_DIR       = BASE_DIR / "outputs"
OUTPUT_PREP      = OUTPUT_DIR / "genes"
OUTPUT_GO        = OUTPUT_DIR / "go"
OUTPUT_GO_REPORT = OUTPUT_DIR / "go_report"

GO_OBO          = BASE_DIR / "inputs" / "go.obo"
GO_ANNOTATIONS  = BASE_DIR / "inputs" / "ccanephora_go.tsv"
GO_JAR          = SCRIPT_DIR / "GOEnrichment.jar"
TRANSCRIPT_GENE = BASE_DIR / "inputs" / "transcript_gene.csv"   # used only for transcriptomics

SPECIES           = ("canephora", "arabica")
VARIATIONS        = ("non_nan", "negative", "positive")
GO_CATEGORIES     = ("bp", "cc", "mf")
P_VALUE_THRESHOLD = 0.01
#DEFENIR NUMERO MINIMO DE GENES PARA CONSIDERAR UM GO TERM SIGNIFICATIVO

def match_transcripts_to_genes(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """Map transcript IDs to gene IDs. Used only when INPUT_TYPE == 'transcriptomics'."""
    mapping = match_df.set_index("transcript")["gene"]
    df["gene"] = df["transcript"].map(mapping)
    df = df.drop(columns="transcript")
    return df


def prep(sp: str, variation: str) -> None:
    input_dir  = INPUT_DATA / sp / variation
    output_dir = OUTPUT_PREP / sp / variation

    # Create input dir if it doesn't exist yet (e.g. negative/positive not yet generated)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [f for f in os.listdir(input_dir) if f != ".DS_Store"]

    if not conditions:
        print(f"  [skip] No input files found for {sp}/{variation} — directory created but empty.")
        return

    if INPUT_TYPE == "transcriptomics":
        transcripts_to_genes = pd.read_csv(TRANSCRIPT_GENE, header=0)

    for i in range(len(conditions)):
        condition_df = pd.read_csv(input_dir / conditions[i], header=0)
        conditions[i] = conditions[i][:-4]

        if INPUT_TYPE == "transcriptomics":
            condition_df.columns = ["transcript"]
            condition_df = match_transcripts_to_genes(condition_df, transcripts_to_genes)
            condition_df["gene"].str.replace("_", "").to_csv(
                output_dir / f"{conditions[i]}.txt", header=False, index=False
            )
        else:
            condition_df.columns = ["gene"]
            condition_df["gene"].str.replace("_", "").to_csv(
                output_dir / f"{conditions[i]}.txt", header=False, index=False
            )


def run_go_enrichment(sp: str, variation: str) -> None:
    input_dir  = OUTPUT_PREP / sp / variation
    output_dir = OUTPUT_GO  / sp / variation
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"  [skip] Gene list directory missing for {sp}/{variation} — skipping GO enrichment.")
        return

    conditions = [f[:-4] for f in os.listdir(input_dir) if f.endswith(".txt")]

    if not conditions:
        print(f"  [skip] No gene list files found for {sp}/{variation} — skipping GO enrichment.")
        return

    log_file   = OUTPUT_GO / "go.log"
    condition_n = 1
    for condition in conditions:
        print(f"Running GO enrichment for condition {condition_n}/{len(conditions)}")
        subp = subprocess.run([
            "java", "-jar", str(GO_JAR),
            "-g",   str(GO_OBO),
            "-a",   str(GO_ANNOTATIONS),
            "-s",   str(input_dir / f"{condition}.txt"),
            "-bpr", str(output_dir / f"go_bp_{condition}.tsv"),
            "-bpg", str(output_dir / f"go_bp_{condition}.svg"),
            "-ccr", str(output_dir / f"go_cc_{condition}.tsv"),
            "-ccg", str(output_dir / f"go_cc_{condition}.svg"),
            "-mfr", str(output_dir / f"go_mf_{condition}.tsv"),
            "-mfg", str(output_dir / f"go_mf_{condition}.svg"),
            "-gf",  "svg"
        ], capture_output=True, text=True, encoding="cp1252", errors="replace")
        print(subp.stdout)
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(str(subp.args) + "\n")
            log.write(subp.stdout + "\n")
            log.write(subp.stderr + "\n")
        condition_n += 1


def parse_condition(condition: str) -> dict[str, str]:
    for prefix in ("go_bp_", "go_cc_", "go_mf_"):
        if condition.startswith(prefix):
            condition = condition[len(prefix):]
            break

    parts = condition.split("__")

    def classify(part: str) -> str:
        if "°C" in part or part.startswith("Rec"):
            return "temperature"
        elif "ppm" in part:
            return "co2"
        else:
            return "water_stress"

    def normalize_temp(part: str) -> str:
        if part.startswith("Rec"):
            return "Rec 14-25/20"
        return part.replace("°C", "").replace("-", "/")

    def normalize_co2(part: str) -> str:
        return part.replace("ppm", "")

    def normalize_ws(part: str) -> str:
        return part.capitalize()

    temps, co2s, wss = [], [], []
    for part in parts:
        cls = classify(part)
        if cls == "temperature":
            temps.append(part)
        elif cls == "co2":
            co2s.append(part)
        else:
            wss.append(part)

    result: dict[str, str] = {}

    if len(temps) == 2:
        result["Temperature"] = f"{normalize_temp(temps[0])} vs {normalize_temp(temps[1])}"
    elif len(temps) == 1:
        result["Temperature"] = normalize_temp(temps[0])
    else:
        raise Exception(f"Cannot determine temperature in: {condition}")

    if len(co2s) == 2:
        result["CO2"] = f"{normalize_co2(co2s[0])} vs {normalize_co2(co2s[1])}"
    elif len(co2s) == 1:
        result["CO2"] = normalize_co2(co2s[0])
    else:
        raise Exception(f"Cannot determine CO2 in: {condition}")

    if len(wss) == 2:
        result["Water stress"] = f"{normalize_ws(wss[0])} vs {normalize_ws(wss[1])}"
    elif len(wss) == 1:
        result["Water stress"] = normalize_ws(wss[0])
    else:
        raise Exception(f"Cannot determine water stress in: {condition}")

    return result


def generate_reports(sp: str, variation: str) -> None:
    input_dir  = OUTPUT_GO        / sp / variation
    output_dir = OUTPUT_GO_REPORT / sp / variation

    if not input_dir.exists():
        print(f"  [skip] GO output directory missing for {sp}/{variation} — skipping report generation.")
        return

    conditions = [c[:-4] for c in os.listdir(input_dir) if c.endswith(".tsv")]

    if not conditions:
        print(f"  [skip] No GO result files found for {sp}/{variation} — skipping report generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions.sort()

    non_significant_all: list[pd.DataFrame] = []

    for category in GO_CATEGORIES:
        filtered_conditions = [c for c in conditions if f"go_{category}_" in c]
        go_aggregate = pd.DataFrame(columns=["GO Term"])
        temperatures = ["Temperature (°C)"]
        co2          = ["CO2 (ppm)"]
        water_stress = ["Water stress"]

        for condition in filtered_conditions:
            go_condition = pd.read_csv(input_dir / f"{condition}.tsv", header=0, sep="\t")
            go_condition["GO Term"] += (" " + go_condition["name"])
            go_condition = go_condition[["GO Term", "p-value"]]
            column_name  = condition.replace(f"C._{sp}.", "").replace(f"go_{category}_", "")

            go_condition = go_condition.rename(columns={"p-value": column_name})
            go_aggregate = go_aggregate.merge(go_condition, on="GO Term", how="outer")

            condition_metadata = parse_condition(condition)
            temperatures.append(condition_metadata["Temperature"])
            co2.append(condition_metadata["CO2"])
            water_stress.append(condition_metadata["Water stress"])

        significant_mask   = go_aggregate.iloc[:, 1:].le(P_VALUE_THRESHOLD).any(axis=1)
        go_significant     = go_aggregate[significant_mask].copy()
        go_non_significant = go_aggregate[~significant_mask].copy()

        # Blank out individual cells that did not meet the threshold —
        # only p-values ≤ P_VALUE_THRESHOLD are kept; the rest become NaN.
        cell_mask = go_significant.iloc[:, 1:].le(P_VALUE_THRESHOLD)
        go_significant.iloc[:, 1:] = go_significant.iloc[:, 1:].where(cell_mask)
        go_non_significant.insert(1, "Ontology", category.upper())
        non_significant_all.append(go_non_significant)

        go_significant.to_csv(output_dir / f"go_{sp}_{category}.csv", index=False)

        if category == GO_CATEGORIES[0]:
            with open(output_dir / f"go_{sp}_headers.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(temperatures)
                writer.writerow(co2)
                writer.writerow(water_stress)

    non_significant_entries = [df for df in non_significant_all if not df.empty]
    if non_significant_entries:
        pd.concat(non_significant_entries, ignore_index=True).to_csv(
            output_dir / f"go_{sp}_uncategorized.csv", index=False
        )


def print_summary() -> None:
    """Print unique gene count and GO annotation coverage across all species and variations."""
    all_genes: set[str] = set()
    for sp in SPECIES:
        for variation in VARIATIONS:
            gene_dir = OUTPUT_PREP / sp / variation
            if not gene_dir.exists():
                continue
            for txt_file in gene_dir.glob("*.txt"):
                try:
                    df = pd.read_csv(txt_file, header=None, dtype=str)
                    if not df.empty:
                        all_genes.update(df.iloc[:, 0].dropna().tolist())
                except pd.errors.EmptyDataError:
                    pass

    annot = pd.read_csv(GO_ANNOTATIONS, sep="\t", header=None, usecols=[0], dtype=str)
    annotated_genes = set(
        annot.iloc[:, 0].str.replace("_", "", regex=False).dropna().tolist()
    )
    covered = all_genes & annotated_genes

    print("\n" + "=" * 50)
    print(f"Input type                                : {INPUT_TYPE}")
    print(f"Unique genes tested across all conditions : {len(all_genes)}")
    print(f"Genes with GO annotations                 : {len(covered)} / {len(all_genes)}")
    print("=" * 50)


if __name__ == "__main__":
    for sp in SPECIES:
        for variation in VARIATIONS:
            prep(sp, variation)
            run_go_enrichment(sp, variation)
            generate_reports(sp, variation)
    print_summary()