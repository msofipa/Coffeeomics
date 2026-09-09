"""
Batch-convert metabolite names to KEGG Compound IDs using the KEGG REST API.

KEGG's /find/compound/<name> endpoint does substring matching against the
compound name and all its listed synonyms, so it's tolerant of common-name
variants (e.g. "Malate" will find "L-Malate"). Results are NOT guaranteed
unique - many primary metabolites have separate KEGG entries for their
stereoisomers (L- vs D-) and, for sugars/sugar-phosphates, their anomeric
form (alpha- vs beta-). This script normalizes names before comparing
(so "Glucose-6-phosphate" matches "D-Glucose 6-phosphate") and, when a
generic non-anomer-specific entry exists alongside alpha/beta variants,
prefers that generic entry - since that's what KEGG pathway maps use.
Anything still ambiguous after that is left for manual review rather
than guessed.

Usage:
    pip install requests --break-system-packages
    python metabolite_get_KEGG_id.py
"""

import json
import re
import time
import csv
from pathlib import Path

import requests

METABOLITES_PATH = Path(r"C:\Users\papan\Desktop\Coffeeomics\metabolomics\metabolomics_index.json")
OUTPUT_PATH = METABOLITES_PATH.parent / "metabolites_kegg_ids.csv"

with open(METABOLITES_PATH, "r") as f:
    METABOLITES = json.load(f)

KEGG_FIND_URL = "https://rest.kegg.jp/find/compound/{}"
REQUEST_DELAY = 0.35  # be polite to the free KEGG API - avoid rate limiting

STEREO_PREFIXES = ("l ", "d ")
ANOMER_PREFIXES = ("alpha ", "beta ")


def normalize(name):
    """Lowercase, collapse hyphens/underscores/extra whitespace to single spaces,
    so 'Glucose-6-phosphate' and 'D-Glucose 6-phosphate' compare equal once
    stereo/anomer prefixes are stripped."""
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def strip_prefixes(name):
    """Remove leading stereo/anomer descriptors so the core compound name
    can be compared regardless of which form was specified."""
    for prefix_set in (ANOMER_PREFIXES, STEREO_PREFIXES):
        for p in prefix_set:
            if name.startswith(p):
                name = name[len(p):]
    return name.strip()


def search_kegg(name):
    """Query KEGG's compound name-search endpoint. Returns list of (id, names_str)."""
    url = KEGG_FIND_URL.format(name)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    hits = []
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        cpd_id, names = line.split("\t")
        cpd_id = cpd_id.replace("cpd:", "")
        hits.append((cpd_id, names))
    return hits


def best_match(query, hits):
    """
    Matching strategy, in order:
    1. Normalized exact match (with or without a D-/L- stereo prefix) on the
       primary (first-listed) name.
    2. Normalized exact match against any synonym.
    3. If multiple core-name matches exist and exactly one has no alpha-/beta-
       anomer prefix, auto-pick that one (it's the generic form KEGG pathway
       maps use) but say so explicitly in the note.
    4. Otherwise, flag for manual review with all candidates listed.
    """
    query_n = normalize(query)
    query_core = strip_prefixes(query_n)

    core_matches = []  # (cpd_id, names, first_name_normalized)
    for cpd_id, names in hits:
        first_name_n = normalize(names.split(";")[0])
        if strip_prefixes(first_name_n) == query_core:
            core_matches.append((cpd_id, names, first_name_n))

    if len(core_matches) == 1:
        cpd_id, names, first_name_n = core_matches[0]
        return cpd_id, names, "exact match on primary name"

    if len(core_matches) > 1:
        non_anomer = [m for m in core_matches if not m[2].startswith(ANOMER_PREFIXES)]
        if len(non_anomer) == 1:
            cpd_id, names, _ = non_anomer[0]
            return cpd_id, names, (
                f"auto-picked generic (non-anomer) form from {len(core_matches)} "
                f"stereo/anomer variants - verify this is the intended form"
            )
        # still ambiguous (e.g. multiple non-anomer stereo variants, or none)
        return None, None, (
            f"{len(core_matches)} equally plausible matches (stereo/anomer variants) "
            f"- review manually"
        )

    # fall back: normalized synonym match
    for cpd_id, names in hits:
        synonyms_n = [normalize(s) for s in names.split(";")]
        if query_n in synonyms_n or query_core in [strip_prefixes(s) for s in synonyms_n]:
            return cpd_id, names, "match on synonym"

    return None, None, "no confident match - review manually"


def main():
    rows = []
    for name in METABOLITES:
        try:
            hits = search_kegg(name)
        except requests.RequestException as e:
            rows.append({"query": name, "kegg_id": "", "matched_name": "",
                         "note": f"request failed: {e}", "all_candidates": ""})
            time.sleep(REQUEST_DELAY)
            continue

        cpd_id, names, note = best_match(name, hits)
        all_candidates = " | ".join(f"{cid}:{n.split(';')[0]}" for cid, n in hits)

        if len(hits) > 1 and "picked" not in note:
            note += f" ({len(hits)} raw candidates - check all_candidates column)"

        rows.append({
            "query": name,
            "kegg_id": cpd_id or "",
            "matched_name": names.split(";")[0] if names else "",
            "note": note,
            "all_candidates": all_candidates,
        })
        time.sleep(REQUEST_DELAY)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "kegg_id", "matched_name", "note", "all_candidates"])
        writer.writeheader()
        writer.writerows(rows)

    unmatched = [r["query"] for r in rows if not r["kegg_id"]]
    print(f"Done. {len(rows) - len(unmatched)}/{len(rows)} matched automatically.")
    if unmatched:
        print("Needs manual review:", ", ".join(unmatched))
    print(f"Full results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()