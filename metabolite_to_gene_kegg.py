"""
Metabolite -> gene lookup table, via the KEGG chain:

    compound (Cxxxxx) --link/reaction--> reaction (Rxxxxx)
                       --link/ko-------> KO (Kxxxxx)
                       --your gene annotation-->  gene(s)

This is a separate, standalone lookup table - NOT the pathway-level
aggregation used in cross_layer_metabolomics_corr.py. That script needs a
coarse, shared feature axis (pathway) for correlation to be meaningful;
this script instead answers a different question - "which specific genes
could plausibly produce or consume this metabolite?" - at the cost of
being noisier for metabolites that sit in many reactions (e.g. Pyruvate,
Glutamate), since those will map to a large, low-specificity gene set.
Directionality (substrate vs. product) is NOT resolved here - a linked
gene means "connected to a reaction involving this compound", not
"produces" or "consumes" it specifically.

Output: one row per (metabolite, reaction, KO, gene) edge - i.e. long
format, so a metabolite touching multiple reactions/genes gets multiple
rows. A per-metabolite gene-count summary is also printed.

──────────────────────────────────────────────────────────────────────────
BEFORE RUNNING - verify GENE_ANNOTATION_PATH / GENE_COL / KO_COL below
match your actual gene functional-annotation file.
──────────────────────────────────────────────────────────────────────────

Usage:
    pip install requests pandas --break-system-packages
    python metabolite_to_gene_kegg.py
"""

import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
METABOLITES_KEGG_IDS_PATH = r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\metabolites_kegg_ids.csv"
OUTPUT_PATH = Path(METABOLITES_KEGG_IDS_PATH).parent / "metabolite_to_gene_via_ko.csv"

# GhostKOALA output: two columns, transcript_id <TAB> KO (blank when no hit).
# Keyed by transcript (GSCOC_T...), so it's joined through TRANSCRIPT_GENE_MAP
# to reach gene level.
USER_KO_PATH = r"C:\Users\papan\Desktop\Coffeeomics\GOA\inputs\user_ko.txt"
TRANSCRIPT_GENE_MAP = r"C:\Users\papan\Desktop\Coffeeomics\GOA\inputs\transcript_gene.csv"

KEGG_LINK_REACTION_URL = "https://rest.kegg.jp/link/reaction/cpd:{}"
KEGG_LINK_KO_URL = "https://rest.kegg.jp/link/ko/rn:{}"
REQUEST_DELAY = 0.35  # be polite to the free KEGG API


# ─────────────────────────────────────────────────────────────────────────
# METABOLITE -> COMPOUND ID
# ─────────────────────────────────────────────────────────────────────────

def load_metabolite_to_kegg_id(kegg_ids_path):
    """{metabolite_name: kegg_id}, excluding rows still needing manual review."""
    df = pd.read_csv(kegg_ids_path)
    resolved = df[df["kegg_id"].notna() & (df["kegg_id"] != "")]
    skipped = df[df["kegg_id"].isna() | (df["kegg_id"] == "")]

    if len(skipped):
        print(f"Skipping {len(skipped)} unresolved metabolite(s) (no kegg_id): "
              f"{skipped['query'].tolist()}")

    return dict(zip(resolved["query"], resolved["kegg_id"]))


# ─────────────────────────────────────────────────────────────────────────
# COMPOUND -> REACTION -> KO  (KEGG REST, cached)
# ─────────────────────────────────────────────────────────────────────────

def fetch_compound_reactions(kegg_id, cache):
    if kegg_id in cache:
        return cache[kegg_id]

    resp = requests.get(KEGG_LINK_REACTION_URL.format(kegg_id), timeout=15)
    resp.raise_for_status()

    reactions = []
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        _, rxn = line.split("\t")
        reactions.append(rxn.replace("rn:", ""))

    cache[kegg_id] = reactions
    time.sleep(REQUEST_DELAY)
    return reactions


def fetch_reaction_kos(reaction_id, cache):
    if reaction_id in cache:
        return cache[reaction_id]

    resp = requests.get(KEGG_LINK_KO_URL.format(reaction_id), timeout=15)
    resp.raise_for_status()

    kos = []
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        _, ko = line.split("\t")
        kos.append(ko.replace("ko:", ""))

    cache[reaction_id] = kos
    time.sleep(REQUEST_DELAY)
    return kos


# ─────────────────────────────────────────────────────────────────────────
# KO -> GENE  (from your existing functional annotation)
# ─────────────────────────────────────────────────────────────────────────

def load_ko_to_genes(user_ko_path, transcript_gene_path):
    """
    Parse GhostKOALA's output (transcript_id <TAB> KO, blank when no hit),
    join through transcript_gene.csv to reach gene level, and invert into
    {ko_id: [gene_id, ...]}.
    """
    transcript_to_gene = pd.read_csv(transcript_gene_path, index_col=0)["gene"]

    ko_to_genes = defaultdict(list)
    with open(user_ko_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            transcript_id = parts[0].strip()
            ko = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            if not ko:
                continue
            gene_id = transcript_to_gene.get(transcript_id)
            if gene_id is not None and pd.notna(gene_id):
                ko_to_genes[ko].append(gene_id)

    return ko_to_genes


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("Loading metabolite -> compound ID map...")
    name_to_kegg_id = load_metabolite_to_kegg_id(METABOLITES_KEGG_IDS_PATH)
    print(f"  {len(name_to_kegg_id)} metabolites resolved to compound IDs.")

    print("Loading gene -> KO annotation from GhostKOALA output...")
    ko_to_genes = load_ko_to_genes(USER_KO_PATH, TRANSCRIPT_GENE_MAP)
    print(f"  {len(ko_to_genes)} distinct KOs found in the annotation.")

    reaction_cache = {}
    ko_cache = {}
    rows = []
    per_metabolite_gene_counts = {}

    for name, kegg_id in name_to_kegg_id.items():
        genes_for_metabolite = set()
        try:
            reactions = fetch_compound_reactions(kegg_id, reaction_cache)
        except requests.RequestException as e:
            print(f"  Warning: reaction lookup failed for {name} ({kegg_id}): {e}")
            continue

        if not reactions:
            rows.append({"metabolite": name, "kegg_id": kegg_id, "reaction": "",
                         "ko": "", "gene": "", "note": "no linked reactions in KEGG"})
            continue

        for rxn in reactions:
            try:
                kos = fetch_reaction_kos(rxn, ko_cache)
            except requests.RequestException as e:
                print(f"  Warning: KO lookup failed for reaction {rxn}: {e}")
                continue

            if not kos:
                rows.append({"metabolite": name, "kegg_id": kegg_id, "reaction": rxn,
                             "ko": "", "gene": "", "note": "no linked KO for this reaction"})
                continue

            for ko in kos:
                genes = ko_to_genes.get(ko, [])
                if not genes:
                    rows.append({"metabolite": name, "kegg_id": kegg_id, "reaction": rxn,
                                 "ko": ko, "gene": "", "note": "no gene in your annotation has this KO"})
                    continue
                for gene in genes:
                    genes_for_metabolite.add(gene)
                    rows.append({"metabolite": name, "kegg_id": kegg_id, "reaction": rxn,
                                 "ko": ko, "gene": gene, "note": ""})

        per_metabolite_gene_counts[name] = len(genes_for_metabolite)

    out_df = pd.DataFrame(rows, columns=["metabolite", "kegg_id", "reaction", "ko", "gene", "note"])
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out_df)} rows to {OUTPUT_PATH}")
    print("\nGenes linked per metabolite (top 10 by count):")
    for name, count in sorted(per_metabolite_gene_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count} gene(s)")


if __name__ == "__main__":
    main()