"""
Batch-convert metabolite names to KEGG Compound IDs using the KEGG REST API.

KEGG's /find/compound/<name> endpoint does substring matching against the
compound name and all its listed synonyms, so it's tolerant of common-name
variants (e.g. "Malate" will find "L-Malate"). Results are NOT guaranteed
unique - many primary metabolites have separate KEGG entries for their
stereoisomers (L- vs D-), so this script surfaces every candidate match
and lets you pick, rather than silently choosing one.

Usage:
    pip install requests --break-system-packages
    python metabolite_to_kegg.py
"""

import json

import requests
import time
import csv

METABOLITES_PATH = R"C:\Users\papan\Desktop\Coffeeomics\metabolomics\metabolomics_index.json"
with open(METABOLITES_PATH, "r") as f:
    METABOLITES = json.load(f)
KEGG_FIND_URL = "https://rest.kegg.jp/find/compound/{}"
REQUEST_DELAY = 0.35  # be polite to the free KEGG API - avoid rate limiting


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
    """Prefer a hit whose FIRST listed name matches the query (KEGG's preferred name),
    since that's usually the canonical, non-stereo-specific-by-accident entry."""
    query_l = query.lower().strip()
    for cpd_id, names in hits:
        first_name = names.split(";")[0].strip().lower()
        if first_name == query_l or first_name == "l-" + query_l or first_name == "d-" + query_l:
            return cpd_id, names, "exact/stereo match on primary name"
    # fall back: any synonym matches exactly
    for cpd_id, names in hits:
        synonyms = [s.strip().lower() for s in names.split(";")]
        if query_l in synonyms:
            return cpd_id, names, "exact match on synonym"
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

        if len(hits) > 1:
            note += f" ({len(hits)} candidates found - check all_candidates column)"

        rows.append({
            "query": name,
            "kegg_id": cpd_id or "",
            "matched_name": names.split(";")[0] if names else "",
            "note": note,
            "all_candidates": all_candidates,
        })
        time.sleep(REQUEST_DELAY)

    with open("metabolites_kegg_ids.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "kegg_id", "matched_name", "note", "all_candidates"])
        writer.writeheader()
        writer.writerows(rows)

    unmatched = [r["query"] for r in rows if not r["kegg_id"]]
    print(f"Done. {len(rows) - len(unmatched)}/{len(rows)} matched automatically.")
    if unmatched:
        print("Needs manual review:", ", ".join(unmatched))
    print("Full results written to metabolites_kegg_ids.csv")


if __name__ == "__main__":
    main()