import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import os
from plyer import notification

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\papan\Desktop\Coffeomics\proteomics"

INPUTS_FOLDER = os.path.join(
    BASE_DIR,
    r"small_dfs_normalized\small_dfs_filtered_normalized_above_10"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "de_raw_results\de_raw_results_50_above10")

OUTPUT_FILE = "de_raw_results"

VALID_FILES = pd.read_csv("valid_contrasts.csv", index_col=0).index

P_VALUE_THRESHOLD = 0.01
LFC_THRESHOLD = 1

REMOVED_SAMPLES = []

# =========================================================
# STORAGE
# =========================================================

final_lfc_df = {}
final_padj_df = {}
final_pval_df = {}

# =========================================================
# DETECT COMPARISON
# =========================================================

def detect_comparison(meta):

    varying_cols = [
        c for c in ["temperature", "co2", "condition"]
        if meta[c].nunique() > 1
    ]

    if len(varying_cols) == 0:
        return None, None

    factor = varying_cols[0]
    levels = sorted(meta[factor].unique())

    return factor, levels


# =========================================================
# RUN DESEQ
# =========================================================

def run_deseq(small_df, comparison_name):

    global final_lfc_df
    global final_padj_df
    global final_pval_df

    print(f"\nRunning: {comparison_name}")

    # -----------------------------------------------------
    # BUILD METADATA
    # -----------------------------------------------------

    meta_data = small_df.columns.to_frame(index=False)

    meta_data.columns = [
        "sample",
        "species",
        "temperature",
        "co2",
        "condition"
    ]

    meta_data.index = meta_data["sample"]

    # -----------------------------------------------------
    # DETECT COMPARISON
    # -----------------------------------------------------

    selected_attribute, levels = detect_comparison(meta_data)

    if selected_attribute is None:
        print(f"Skipping {comparison_name}: no varying factor")
        return

    print(f"Factor: {selected_attribute}")
    print(f"Levels: {levels}")

    # -----------------------------------------------------
    # COUNTS
    # -----------------------------------------------------

    counts = small_df.copy()

    counts.columns = meta_data.index

    counts = counts.round().astype(int)

    # DEBUG 1
    print("\n[DEBUG] counts shape:", counts.shape)
    print("[DEBUG] meta_data shape:", meta_data.shape)

    # -----------------------------------------------------
    # PREFILTER LOW COUNTS
    # -----------------------------------------------------

    counts = counts[counts.sum(axis=1) >= 10]

    if counts.shape[0] == 0:
        print(f"Skipping {comparison_name}: no genes after filtering")
        return

    print(f"Genes after filtering: {counts.shape[0]}")

    # DEBUG 2
    print(
        "\n[DEBUG] counts columns == meta index:",
        all(counts.columns == meta_data.index)
    )

    # -----------------------------------------------------
    # CHECK REPLICATES
    # -----------------------------------------------------

    group_counts = meta_data[selected_attribute].value_counts()

    print("\nReplicates per group:")
    print(group_counts)

    if (group_counts < 2).any():
        print(f"Skipping {comparison_name}: insufficient replicates")
        return

    # -----------------------------------------------------
    # CRITICAL DEBUG CHECKS
    # -----------------------------------------------------

    print(
        "\n[DEBUG] Unique levels in factor:",
        meta_data[selected_attribute].unique()
    )

    print(
        "[DEBUG] Factor dtype:",
        meta_data[selected_attribute].dtype
    )

    # enforce categorical
    meta_data[selected_attribute] = (
        meta_data[selected_attribute]
        .astype("category")
    )

    # align metadata
    meta_data = meta_data.loc[counts.columns]

    # final alignment check
    assert all(counts.columns == meta_data.index), \
        "ERROR: counts and metadata are misaligned"

    print("[DEBUG] FINAL alignment check PASSED")

    # -----------------------------------------------------
    # BUILD DESEQ DATASET
    # -----------------------------------------------------

    dds = DeseqDataSet(
        counts=counts.T,
        metadata=meta_data,
        design=f"~{selected_attribute}"
    )

    # DEBUG 3
    print("\n[DEBUG] dds counts shape:", dds.X.shape)
    print("[DEBUG] dds obs shape:", dds.obs.shape)
    print("[DEBUG] design formula:", dds.design)

    # -----------------------------------------------------
    # FIT MODEL
    # -----------------------------------------------------

    print("\nFitting DESeq2 model...")

    dds.fit_size_factors()

    dds.fit_genewise_dispersions()

    dds.fit_dispersion_trend()

    dds.fit_dispersion_prior()

    dds.fit_MAP_dispersions()

    dds.refit_cooks = False

    dds.fit_LFC()

    # -----------------------------------------------------
    # CONTRAST
    # -----------------------------------------------------

    contrast = [
        selected_attribute,
        levels[1],
        levels[0]
    ]

    print("\nContrast:", contrast)

    # -----------------------------------------------------
    # STATS
    # -----------------------------------------------------

    ds = DeseqStats(
        dds,
        contrast=contrast,
        alpha=P_VALUE_THRESHOLD,
        cooks_filter=False,
        independent_filter=False
    )

    print("\nRunning Wald test...")

    ds.run_wald_test()

    ds._p_value_adjustment()

    ds.summary()

    results = ds.results_df.copy()

    # -----------------------------------------------------
    # EXTRACT RESULTS
    # -----------------------------------------------------

    lfc = results["log2FoldChange"]

    padj = results["padj"]

    pval = results["pvalue"]

    # -----------------------------------------------------
    # FILTERING
    # -----------------------------------------------------

    valid = (~lfc.isna()) & (~padj.isna())

    significant = valid & (padj <= P_VALUE_THRESHOLD)

    abs_lfc = lfc.abs()

    mask = (~significant) | (abs_lfc <= LFC_THRESHOLD)

    filtered_lfc = lfc.mask(mask)

    filtered_padj = padj.mask(mask)

    # -----------------------------------------------------
    # STORE FILTERED RESULTS
    # -----------------------------------------------------

    final_lfc_df[comparison_name] = filtered_lfc

    final_padj_df[comparison_name] = filtered_padj

    final_pval_df[comparison_name] = pval

    # -----------------------------------------------------
    # DEBUG FILTERING
    # -----------------------------------------------------

    print("\n[DEBUG] Significant proteins:", significant.sum())

    print(
        "[DEBUG] Proteins passing LFC threshold:",
        (abs_lfc > LFC_THRESHOLD).sum()
    )

    print(
        "[DEBUG] Final DE proteins:",
        (~mask).sum()
    )

    print(f"\nStored results for: {comparison_name}")


# =========================================================
# MAIN LOOP
# =========================================================

for file in os.listdir(INPUTS_FOLDER):

    file_path = os.path.join(INPUTS_FOLDER, file)

    if not file.endswith(".csv"):
        continue

    print("\n=================================================")
    print(f"Processing file: {file}")
    print("=================================================")

    if file in VALID_FILES:

        small_df = pd.read_csv(
            file_path,
            index_col=0,
            header=[0, 1, 2, 3, 4]
        )

        if len(REMOVED_SAMPLES) > 0:

            keep_cols = [
                c for c in small_df.columns
                if c[0] not in REMOVED_SAMPLES
            ]

            small_df = small_df[keep_cols]

        print(f"Shape: {small_df.shape}")

        comparison_name = pd.read_csv("valid_contrasts.csv", index_col=0).loc[file]["combination"]

        try:

            run_deseq(
                small_df,
                comparison_name
            )

        except Exception as e:

            print(f"\nERROR in {comparison_name}")
            print(str(e))


# =========================================================
# BUILD FINAL DATAFRAMES
# =========================================================

print("\n=================================================")
print("BUILDING FINAL DATAFRAMES")
print("=================================================")

final_lfc_df = (
    pd.DataFrame(final_lfc_df)
    .reset_index()
    .rename(columns={"index": "protein"})
)

final_padj_df = (
    pd.DataFrame(final_padj_df)
    .reset_index()
    .rename(columns={"index": "protein"})
)

final_pval_df = (
    pd.DataFrame(final_pval_df)
    .reset_index()
    .rename(columns={"index": "protein"})
)

print("LFC df shape:", final_lfc_df.shape)

print("Padj df shape:", final_padj_df.shape)

print("Pval df shape:", final_pval_df.shape)

# =========================================================
# SAVE OUTPUT
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

notification.notify(
    title="DESeq2",
    message="Run Finished!"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f"{OUTPUT_FILE}_de_raw_results.xlsx"
)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    final_lfc_df.to_excel(
        writer,
        sheet_name="log2FoldChange",
        index=False
    )

    final_padj_df.to_excel(
        writer,
        sheet_name="padj_values",
        index=False
    )

    final_pval_df.to_excel(
        writer,
        sheet_name="p_values",
        index=False
    )

print(f"\nSaved results to:\n{output_file}")