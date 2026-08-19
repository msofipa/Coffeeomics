"""
make_correlations.py
─────────────────────
Library of functions for building Pearson correlation matrices (within-layer
and cross-layer) between proteomics/transcriptomics samples in the Coffeeomics
project, with NaN-aware hierarchical clustering and annotated heatmap output.

Called from make_correlations.ipynb. Public entry points:
    - build_and_save(...)         cross-layer (proteomics vs transcriptomics), asymmetric matrix
    - build_and_save_single(...)  within-layer (one omics vs itself), symmetric matrix

Each entry point saves 8 figures (PNG + PDF) per species:
    1. {species}_pearson_tree                  - dendrogram
    2. {species}_pearson_clustermap             - clustered r-value heatmap
    3. {species}_pearson_annotated_clustermap   - clustermap + Temp/CO2/Water side-bars
    4. {species}_pearson_ordered_heatmap        - r-value heatmap, numeric sample order
    5. {species}_gene_overlap_clustermap        - clustered heatmap of gene counts per cell
    6. {species}_gene_overlap_ordered_heatmap   - same, numeric sample order

NOTE: this is a cleaned version - the original file had each builder function
defined twice (an artifact of notebook export); only the single active
definition of each is kept here.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from pathlib import Path

def safe_name(text):  
    # Remove anything that isn't a word character (a-z, 0-9, _), whitespace, or hyphen
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace any run of whitespace with a single underscore, and strip leading/trailing spaces
    text = re.sub(r"\s+", "_", text.strip())
    return text
    # e.g. "C. arabica | 25°C" → "C_arabica__25C"

def tokenise(text):
    # Find every contiguous run of letters or digits — punctuation/spaces are discarded
    return re.findall(r'[A-Za-z0-9]+', text)
    # e.g. "7A | 25/20°C" → ["7A", "25", "20", "C"]

def col_to_species(col, short_id_to_species):
    tokens = tokenise(col)
    for sid, sp in short_id_to_species.items():
        if sid in tokens:
            return sp
    return None

def label_to_group(full_label):
    # Tokenise the label so we can do clean substring matching
    tokens = tokenise(full_label)

    # Walk every contrast group and its associated short IDs (e.g. "7A", "8B")
    for group, short_ids in contrast_groups.items():
        for sid in short_ids:
            # If this short ID appears as a token in the label, this is the right group
            if sid in tokens:
                return group

    # No group matched → return None
    return None

import json

# Global consultado por label_to_group() e make_display_label().
# É atualizado por set_contrast_groups() / set_contrast_groups_merged() —
# as funções build_and_save / build_and_save_single /
# build_single_layer_tree_and_heatmaps já tratam disto internamente
# a partir do(s) argumento(s) `layer` / `layers`.
contrast_groups = {}


def load_contrast_groups(layer, base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Carrega o contrast_groups.json correspondente a uma camada específica.

    Parameters
    ----------
    layer : str
        Uma de: "transcriptomics", "proteomics", "metabolomics"
    base_dir : str
        Diretório base onde estão as pastas de cada camada.
    """
    valid_layers = {"transcriptomics", "proteomics", "metabolomics"}
    if layer not in valid_layers:
        raise ValueError(f"layer deve ser um de {valid_layers}, recebido: {layer}")

    file_path = Path(base_dir) / layer / f"contrast_groups_{layer}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    return loaded


def set_contrast_groups(layer, base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Carrega o contrast_groups.json de UMA camada e atualiza o global
    `contrast_groups` usado por label_to_group() / make_display_label().
    """
    global contrast_groups
    contrast_groups = load_contrast_groups(layer, base_dir=base_dir)
    return contrast_groups


def set_contrast_groups_merged(layers, base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Carrega o contrast_groups.json de VÁRIAS camadas (ex.: proteomics +
    transcriptomics, para comparações cross-layer) e faz merge num único
    dicionário, unindo os short_ids de cada grupo. Atualiza o global
    `contrast_groups`.

    Parameters
    ----------
    layers : list[str] | tuple[str]
        Camadas a combinar, ex.: ("proteomics", "transcriptomics")
    """
    global contrast_groups
    merged = {}
    for layer in layers:
        layer_groups = load_contrast_groups(layer, base_dir=base_dir)
        for group, short_ids in layer_groups.items():
            merged.setdefault(group, [])
            merged[group] = list(dict.fromkeys(merged[group] + list(short_ids)))
    contrast_groups = merged
    return contrast_groups

def make_display_label(full_col):
    tokens = tokenise(full_col)

    for group, short_ids in contrast_groups.items():
        for sid in short_ids:
            if sid in tokens:
                # group looks like "C. arabica | 25/20°C | 380ppm | without"
                parts = group.split(" | ")
                # Drop parts[0] (species), keep the experimental conditions
                condition = " | ".join(parts[1:])
                # e.g. → "7A | 25/20°C | 380ppm | without"
                return f"{sid} | {condition}"

    # No match found — return the original column name unchanged
    return full_col

def group_to_attributes(group_key):
    # Split on " | " to get the four components of the group key
    # Expected format: "C. arabica | 25/20°C | 380ppm | without"
    parts = group_key.split(" | ")
    # parts[0] = species, parts[1] = temperature, parts[2] = CO2, parts[3] = water stress

    # Guard: if the format is unexpected, return safe NA defaults
    if len(parts) < 4:
        return {"Temperature": "NA", "CO2_ppm": "NA", "Water_Stress": "NA"}

    return {
        "Temperature":  parts[1],   # e.g. "25/20°C"
        "CO2_ppm":      parts[2],   # e.g. "380ppm"
        "Water_Stress": parts[3],   # e.g. "without"
    }

import sys
from pathlib import Path

sys.path.append(
    str(Path(r"C:\Users\papan\Desktop\Coffeeomics"))
)

from color_maps import COLOR_MAPS

def build_annotation_matrices(display_labels, original_labels):
    annotation_matrices = {}  # will hold one DataFrame per attribute
    annotation_labels   = {}  # will hold the raw value strings per attribute

    for attribute in ["Temperature", "CO2_ppm", "Water_Stress"]:
        label_to_color = COLOR_MAPS[attribute]  # dict mapping value → hex color
        colors = []
        labels = []

        for orig in original_labels:
            # Find which contrast group this sample belongs to
            grp = label_to_group(orig)
            if grp:
                # Extract the value for this attribute from the group key
                attrs = group_to_attributes(grp)
                val   = attrs.get(attribute, "NA")
            else:
                val = "NA"

            # Look up the hex color for this value; fall back to black if unknown
            colors.append(label_to_color.get(val, "#000000"))
            labels.append(val)

        # Store as a single-column DataFrame indexed by display label
        # (this is the format seaborn clustermap expects for row_colors/col_colors)
        annotation_matrices[attribute] = pd.DataFrame(
            colors,
            index=display_labels,
            columns=[attribute]
        )
        annotation_labels[attribute] = labels

    return annotation_matrices, annotation_labels

def save_figure(fig_or_grid, base_path):
    for ext in (".png", ".pdf"):
        fig_or_grid.savefig(
            base_path.with_suffix(ext),  # swap/add the extension on the Path object
            dpi=300,                      # high resolution
            bbox_inches="tight"           # crop whitespace around the figure
        )
    print(f"  Saved: {base_path}.png / .pdf")

def color_clustermap_labels(clustergrid, original_labels, label_to_group_map, group_to_color):
    # Build a lookup: display label → contrast group
    display_to_group = {
        make_display_label(lbl): label_to_group_map[lbl]
        for lbl in original_labels
    }

    # Color y-axis (row) tick labels
    for tick in clustergrid.ax_heatmap.get_yticklabels():
        grp = display_to_group.get(tick.get_text())
        if grp:
            tick.set_color(group_to_color.get(grp, "black"))

    # Color x-axis (column) tick labels
    for tick in clustergrid.ax_heatmap.get_xticklabels():
        grp = display_to_group.get(tick.get_text())
        if grp:
            tick.set_color(group_to_color.get(grp, "black"))

def build_annotated_heatmap(mat_labeled, row_lkg, col_lkg, original_labels,
                             display_labels, sp, comparison_folder,
                             unique_groups, group_to_color, COLOR_MAPS,
                             FIRST_INPUT_DEF, SECOND_INPUT_DEF):
    """
    Builds and saves a seaborn clustermap with:
      - Pre-computed row/column linkage (no re-clustering)
      - Color sidebars for Temperature, CO2_ppm, and Water_Stress
      - Colored tick labels matching contrast groups
      - One legend per attribute + one contrast-group legend

    Parameters
    ----------
    mat_labeled       : pd.DataFrame — correlation matrix with display labels as index/columns
    row_lkg           : np.ndarray   — pre-computed row linkage matrix
    col_lkg           : np.ndarray   — pre-computed column linkage matrix
    original_labels   : list[str]    — raw column names (used for group lookup)
    display_labels    : list[str]    — human-readable labels (used on plot axes)
    sp                : str          — species name (used in title and filename)
    comparison_folder : Path         — folder where output files are saved
    unique_groups     : list[str]    — all contrast group keys in order
    group_to_color    : dict         — contrast group key → RGB color tuple
    COLOR_MAPS        : dict         — attribute name → {value: hex color}
    FIRST_INPUT_DEF   : str          — label for the row axis (proteomics)
    SECOND_INPUT_DEF  : str          — label for the column axis (transcriptomics)
    """

    # ── 1. Build color sidebar matrices ───────────────────────────────────────
    # Returns one single-column DataFrame per attribute (Temperature, CO2_ppm, Water_Stress)
    # Each DataFrame maps display_labels → hex color string
    annotation_matrices, annotation_labels = build_annotation_matrices(
        display_labels, original_labels
    )

    attributes = ["Temperature", "CO2_ppm", "Water_Stress"]

    # ── 2. Build group map for tick-label coloring ────────────────────────────
    # Done before clustermap so it's available for both coloring and the legend
    label_to_group_map = {lbl: label_to_group(lbl) for lbl in original_labels}

    # Find which contrast groups are actually present in this species' samples
    # (avoids adding legend entries for groups that don't appear in the plot)
    sp_groups = {
        grp for grp in label_to_group_map.values()
        if grp is not None
    }

    # ── 3. Draw the clustermap ────────────────────────────────────────────────
    cg = sns.clustermap(
        mat_labeled,
        row_linkage=row_lkg,            # use pre-computed linkage — no re-clustering
        col_linkage=col_lkg,
        cmap="PuOr",                    # diverging colormap: purple = negative, orange = positive
        center=0,                       # anchor neutral color at r = 0
        figsize=(36, 36),
        linewidths=0.3,                 # thin gridlines between cells
        annot=True,                     # print r values inside each cell
        fmt=".2f",
        annot_kws={"color": "white", "fontsize": 4},
        # One color strip per attribute on both axes
        row_colors=[
            annotation_matrices[a].iloc[:, 0].map(lambda c: c)
            for a in attributes
        ],
        col_colors=[
            annotation_matrices[a].iloc[:, 0].map(lambda c: c)
            for a in attributes
        ],
    )

    # ── 4. Title and axis labels ──────────────────────────────────────────────
    cg.fig.suptitle(
        f"{sp} — Annotated Pearson Clustermap\n"
        f"{FIRST_INPUT_DEF}  vs  {SECOND_INPUT_DEF}",
        y=1.02,
        fontsize=15
    )
    cg.ax_heatmap.set_xlabel(SECOND_INPUT_DEF, fontsize=15, labelpad=12)
    cg.ax_heatmap.set_ylabel(FIRST_INPUT_DEF,  fontsize=15, labelpad=12)

    # ── 5. Color tick labels by contrast group ────────────────────────────────
    # Must be called after the figure is drawn so tick labels exist
    color_clustermap_labels(cg, original_labels, label_to_group_map, group_to_color)

    # ── 6. Per-attribute legends ──────────────────────────────────────────────
    # One legend per attribute, stacked vertically on the right side of the figure
    # cg.fig.add_artist(leg) is required when adding more than one legend —
    # otherwise each new call to cg.fig.legend() replaces the previous one
    for i, attr in enumerate(attributes):
        handles = [
            mpatches.Patch(facecolor=color, label=val)
            for val, color in COLOR_MAPS[attr].items()
        ]
        leg = cg.fig.legend(
            handles=handles,
            title=attr.replace("_", " "),
            loc="upper left",
            bbox_to_anchor=(1.25, 0.90 - i * 0.20),   # stack downward with 0.20 spacing
            fontsize=15,
            title_fontsize=15,
            frameon=True
        )
        cg.fig.add_artist(leg)

    # ── 7. Contrast-group legend ──────────────────────────────────────────────
    # Shows only groups present in this species — placed at the top right
    group_handles = [
        mpatches.Patch(
            facecolor=group_to_color[g],
            label=" | ".join(g.split(" | ")[1:])   # drop species prefix from label
        )
        for g in unique_groups
        if g in sp_groups
    ]
    cg.fig.legend(
        handles=group_handles,
        title="Contrast group",
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    # ── 8. Save and close ─────────────────────────────────────────────────────
    save_figure(cg, comparison_folder / f"{safe_name(sp)}_pearson_annotated_clustermap")
    plt.close()   # release memory — critical when looping over multiple species

def nan_aware_condensed_distances(mat_values, THRESHOLD_OVERLAP=10):
    """
    Computes a condensed correlation-distance matrix (1 − r) between rows,
    ignoring NaN values pairwise.
    Returns a 1-D condensed array for scipy.cluster.hierarchy.linkage.
    """
    n = mat_values.shape[0]
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            row_i = mat_values[i]
            row_j = mat_values[j]
            mask = ~np.isnan(row_i) & ~np.isnan(row_j)

            if mask.sum() >= THRESHOLD_OVERLAP:
                r, _ = pearsonr(row_i[mask], row_j[mask])
                dist[i, j] = dist[j, i] = 1 - np.clip(r, -1, 1)
            else:
                # Not enough overlap → treat as maximally dissimilar
                dist[i, j] = dist[j, i] = 2.0  # max possible (1 − (−1))

    return squareform(dist, checks=False)  # → condensed 1-D array

def _sort_key(s):
    m = re.match(r'^(\d+)([A-Za-z]*)$', s)
    return (int(m.group(1)), m.group(2).upper()) if m else (float('inf'), s)


def build_and_save(sp, sp_samples, prot_aligned, trans_aligned, comparison_folder,
                   unique_groups, group_to_color, COLOR_MAPS,
                   FIRST_INPUT_DEF, SECOND_INPUT_DEF, show_attribute_strips=True,
                   layers=("proteomics", "transcriptomics"),
                   contrast_groups_base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Cross-layer Pearson correlation between proteomics and transcriptomics.
    Rows = prot samples, columns = trans samples. Matrix is asymmetric.
    Produces six outputs in comparison_folder.

    Parameters
    ----------
    sp                        : str          — species name
    sp_samples                : list[str]    — sample column names belonging to this species
    prot_aligned              : pd.DataFrame — proteomics layer (genes × samples)
    trans_aligned              : pd.DataFrame — transcriptomics layer (genes × samples)
    comparison_folder         : Path         — output folder for saved figures
    unique_groups             : list[str]    — all contrast group keys in order
    group_to_color            : dict         — contrast group key → RGB color tuple
    COLOR_MAPS                : dict         — attribute name → {value: hex color}
    FIRST_INPUT_DEF            : str          — row axis label (proteomics)
    SECOND_INPUT_DEF           : str          — column axis label (transcriptomics)
    layers                     : tuple[str]   — layers whose contrast_groups_<layer>.json
                                                 get merged (short_ids unioned per group)
                                                 via set_contrast_groups_merged() before
                                                 building. Set to None to skip and use
                                                 whatever contrast_groups is already set.
    contrast_groups_base_dir  : str          — base_dir passed through to load_contrast_groups
    """

    if layers is not None:
        set_contrast_groups_merged(layers, base_dir=contrast_groups_base_dir)

    print(f"\nBuilding matrix for {sp} ({len(sp_samples)} samples)...")

    # ── 1. Pearson cross-layer correlation matrix ──────────────────────────────
    # Asymmetric: cell [s1, s2] = Pearson r between prot[s1] and trans[s2]
    # across shared non-NaN genes. Diagonal [s,s] = within-sample cross-layer r.
    mat            = pd.DataFrame(np.nan, index=sp_samples, columns=sp_samples)
    n_mat          = pd.DataFrame(0,      index=sp_samples, columns=sp_samples)
    excluded_pairs = []

    for s1 in sp_samples:
        for s2 in sp_samples:
            v1      = prot_aligned[s1]
            v2      = trans_aligned[s2]
            mask    = v1.notna() & v2.notna()
            n_valid = mask.sum()
            n_mat.loc[s1, s2] = n_valid
            if n_valid >= 3:
                r, _ = pearsonr(v1[mask], v2[mask])
                mat.loc[s1, s2] = r
            else:
                excluded_pairs.append((s1, s2, int(n_valid)))

    if excluded_pairs:
        print(f"  ⚠  {len(excluded_pairs)} comparison(s) excluded "
              f"(fewer than 3 overlapping genes):")
        for s1, s2, n in excluded_pairs:
            print(f"      {s1}  vs  {s2}  →  {n} gene(s) in common")
    else:
        print(f"  ✓  No comparisons excluded for {sp}.")

    # ── 1b. Colormaps ──────────────────────────────────────────────────────────
    import copy
    cmap_bad   = copy.copy(plt.get_cmap("PuOr"))
    cmap_bad.set_bad(color="black")

    cmap_genes = copy.copy(plt.get_cmap("Blues"))
    cmap_genes.set_bad(color="black")

    # ── 2. Hierarchical clustering (NaN-aware) ─────────────────────────────────
    # Matrix is asymmetric so row and column linkages are computed separately
    row_lkg = linkage(nan_aware_condensed_distances(mat.values),   method="average")
    col_lkg = linkage(nan_aware_condensed_distances(mat.values.T), method="average")

    # ── 3. Labels and per-sample colors ───────────────────────────────────────
    original_labels    = mat.index.tolist()
    display_labels     = [make_display_label(s) for s in original_labels]
    label_to_group_map = {lbl: label_to_group(lbl) for lbl in original_labels}

    label_colors = [
        group_to_color.get(label_to_group_map[lbl], "black")
        for lbl in original_labels
    ]

    sp_groups = {
        grp for grp in label_to_group_map.values()
        if grp is not None
    }
    legend_handles = [
        mpatches.Patch(
            facecolor=group_to_color[g],
            label=" | ".join(g.split(" | ")[1:])
        )
        for g in unique_groups
        if g in sp_groups
    ]

    #{sp}_pearson_tree
    # ── 4. Dendrogram (with optional Temperature/CO2/Water-Stress strips) ───────
    attributes = ["Temperature", "CO2_ppm", "Water_Stress"]  # order of strips, closest-to-tree first

    if show_attribute_strips:
        annotation_matrices, _ = build_annotation_matrices(display_labels, original_labels)
        n_attrs = len(attributes)
        fig, axes = plt.subplots(
            1, 2 + n_attrs,   # tree, spacer, then one axis per strip
            figsize=(14 + 0.8 * n_attrs, 32),
            gridspec_kw={
                "width_ratios": [10, 8] + [0.35] * n_attrs,  # 0.5 = spacer width — tune this
                "wspace": 0.05,
            },
        )
        ax = axes[0]
        spacer_ax = axes[1]
        spacer_ax.axis("off")   # invisible — just holds space
        strip_axes = list(axes[2:])
    else:
        fig, ax = plt.subplots(figsize=(8, 32))
        strip_axes = []

    ddata = dendrogram(
        row_lkg,
        labels=display_labels,
        orientation="left",
        leaf_rotation=0,
        leaf_font_size=15,
        ax=ax
    )

    leaf_order     = ddata["leaves"]
    ordered_colors = [label_colors[i] for i in leaf_order]
    for tick_label, color in zip(ax.get_yticklabels(), ordered_colors):
        tick_label.set_color(color)

    # ── attribute color strips (mirrors build_annotated_heatmap's row_colors) ───
    if show_attribute_strips:
        y_positions = ax.get_yticks()  # same bottom-to-top order as leaf_order (verified empirically)

        for strip_ax, attr in zip(strip_axes, attributes):
            colors_ordered = [annotation_matrices[attr].iloc[i, 0] for i in leaf_order]
            for y, color in zip(y_positions, colors_ordered):
                strip_ax.add_patch(
                    plt.Rectangle((0, y - 5), 1, 10, facecolor=color, edgecolor="white")
                )
            strip_ax.set_ylim(ax.get_ylim())
            strip_ax.set_xlim(0, 1)
            strip_ax.set_xticks([])
            strip_ax.set_yticks([])
            strip_ax.set_title(attr.replace("_", " "), fontsize=15, rotation=90, va="bottom")
            for spine in strip_ax.spines.values():
                spine.set_visible(False)

    ax.set_title(
        f"{sp} — Pearson Correlation Tree\n"
        f"{FIRST_INPUT_DEF}  vs  {SECOND_INPUT_DEF}",
        fontsize=15
    )
    ax.set_ylabel("Samples", fontsize=15)
    ax.set_xlabel("Pearson distance  (1 − r)", fontsize=15)

    # ── legends ───────────────────────────────────────────────────────────────
    # Contrast-group legend — always shown, anchored to the whole figure so its
    # position stays consistent whether or not the strips are present
    leg_group = fig.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(-0.2, 0.8),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )
    fig.add_artist(leg_group)

    # One extra legend per attribute strip, stacked below the contrast-group legend
    if show_attribute_strips:
        for i, attr in enumerate(attributes):
            handles = [
                mpatches.Patch(facecolor=color, label=val)
                for val, color in COLOR_MAPS[attr].items()
            ]
            leg = fig.legend(
                handles=handles,
                title=attr.replace("_", " "),
                loc="upper left",
                bbox_to_anchor=(-0.2, 0.6 - i * 0.08),
                fontsize=15,
                title_fontsize=15,
                frameon=True
            )
            fig.add_artist(leg)

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_tree")
    plt.close()

    # ── 5. Standard clustermap ─────────────────────────────────────────────────
    mat_labeled         = mat.copy()
    mat_labeled.index   = display_labels
    mat_labeled.columns = display_labels

    annot_matrix = mat_labeled.map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )

    clustergrid = sns.clustermap(
        mat_labeled,
        row_linkage=row_lkg,
        col_linkage=col_lkg,
        cmap=cmap_bad,
        center=0,
        figsize=(32, 32),
        linewidths=0.5,
        annot=annot_matrix,
        fmt="",
        annot_kws={"color": "white", "fontsize": 5}
    )

    clustergrid.fig.suptitle(
        f"{sp} — Pearson Clustermap\n"
        f"Rows = {FIRST_INPUT_DEF}\nColumns = {SECOND_INPUT_DEF}",
        y=1.02,
        fontsize=15
    )
    clustergrid.ax_heatmap.set_xlabel(SECOND_INPUT_DEF, fontsize=15, labelpad=12)
    clustergrid.ax_heatmap.set_ylabel(FIRST_INPUT_DEF,  fontsize=15, labelpad=12)

    color_clustermap_labels(clustergrid, original_labels, label_to_group_map, group_to_color)

    clustergrid.ax_heatmap.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    save_figure(clustergrid, comparison_folder / f"{safe_name(sp)}_pearson_clustermap")
    plt.close()

    # ── 6. Annotated clustermap ────────────────────────────────────────────────
    build_annotated_heatmap(
        mat_labeled=mat_labeled,
        row_lkg=row_lkg,
        col_lkg=col_lkg,
        original_labels=original_labels,
        display_labels=display_labels,
        sp=sp,
        comparison_folder=comparison_folder,
        unique_groups=unique_groups,
        group_to_color=group_to_color,
        COLOR_MAPS=COLOR_MAPS,
        FIRST_INPUT_DEF=FIRST_INPUT_DEF,
        SECOND_INPUT_DEF=SECOND_INPUT_DEF
    )

    # ── 7. Ordered heatmap ─────────────────────────────────────────────────────
    ordered_samples      = sorted(sp_samples, key=_sort_key)
    ordered_display      = [make_display_label(s) for s in ordered_samples]
    ordered_disp_to_orig = dict(zip(ordered_display, ordered_samples))

    mat_ordered         = mat.loc[ordered_samples, ordered_samples].copy()
    mat_ordered.index   = ordered_display
    mat_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(28, 24))

    sns.heatmap(
        mat_ordered,
        ax=ax,
        cmap=cmap_bad,
        center=0,
        linewidths=0.4,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        annot=True,                              # print the value in each cell
        fmt=".2f",                                # 2 decimal places, e.g. "0.87"
        annot_kws={"color": "white", "fontsize": 6},  # force white text regardless of cell shade
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(15)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(15)

    ax.set_title(
        f"{sp} — Pearson r  (numeric / alphabetic order)\n"
        f"Rows = {FIRST_INPUT_DEF}\nColumns = {SECOND_INPUT_DEF}",
        fontsize=15,
    )
    ax.set_xlabel(SECOND_INPUT_DEF, fontsize=15, labelpad=12)
    ax.set_ylabel(FIRST_INPUT_DEF,  fontsize=15, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_ordered_heatmap")
    plt.close()
    
    # ── 8. Gene overlap heatmap ────────────────────────────────────────────────
    # Shows how many genes contributed to each Pearson r. Excluded pairs (< 3)
    # still appear with their real count so you can see exactly why they failed.
    n_labeled         = n_mat.copy()
    n_labeled.index   = display_labels
    n_labeled.columns = display_labels

    n_annot = n_labeled.map(lambda x: str(int(x)) if pd.notna(x) else "")

    # 8a. Clustermap — same row/col ordering as the correlation clustermap
    n_clustergrid = sns.clustermap(
        n_labeled,
        row_linkage=row_lkg,
        col_linkage=col_lkg,
        cmap=cmap_genes,
        figsize=(32, 32),
        linewidths=0.5,
        annot=n_annot,
        fmt="",
        annot_kws={"fontsize": 5}
    )

    n_clustergrid.fig.suptitle(
        f"{sp} — Genes used per correlation\n"
        f"Rows = {FIRST_INPUT_DEF}\nColumns = {SECOND_INPUT_DEF}",
        y=1.02,
        fontsize=15
    )
    n_clustergrid.ax_heatmap.set_xlabel(SECOND_INPUT_DEF, fontsize=10, labelpad=12)
    n_clustergrid.ax_heatmap.set_ylabel(FIRST_INPUT_DEF,  fontsize=10, labelpad=12)

    color_clustermap_labels(n_clustergrid, original_labels, label_to_group_map, group_to_color)

    n_clustergrid.ax_heatmap.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    save_figure(n_clustergrid, comparison_folder / f"{safe_name(sp)}_gene_overlap_clustermap")
    plt.close()

    # 8b. Ordered heatmap — same numeric/alphabetic ordering as section 7
    n_ordered         = n_mat.loc[ordered_samples, ordered_samples].copy()
    n_ordered.index   = ordered_display
    n_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(28, 24))

    sns.heatmap(
        n_ordered,
        ax=ax,
        cmap=cmap_genes,
        linewidths=0.4,
        linecolor="lightgrey",
        annot=n_ordered.map(lambda x: str(int(x)) if pd.notna(x) else ""),
        fmt="",
        annot_kws={"fontsize": 5},
        cbar_kws={"shrink": 0.6, "label": "genes used"},
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(15)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(15)

    ax.set_title(
        f"{sp} — Genes used per correlation  (numeric / alphabetic order)\n"
        f"Rows = {FIRST_INPUT_DEF}\nColumns = {SECOND_INPUT_DEF}",
        fontsize=15,
    )
    ax.set_xlabel(SECOND_INPUT_DEF, fontsize=10, labelpad=12)
    ax.set_ylabel(FIRST_INPUT_DEF,  fontsize=10, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_gene_overlap_ordered_heatmap")
    plt.close()

def build_and_save_single(sp, sp_samples, data, comparison_folder,
                          unique_groups, group_to_color, COLOR_MAPS, INPUT_DEF,
                          layer=None, contrast_groups_base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Single-layer variant of build_and_save.
    Correlates one dataset's samples against themselves (symmetric matrix).
    Produces six outputs in comparison_folder.

    Parameters
    ----------
    sp                        : str          — species name
    sp_samples                : list[str]    — sample column names belonging to this species
    data                      : pd.DataFrame — single omics layer (genes x samples)
    comparison_folder         : Path         — output folder for saved figures
    unique_groups             : list[str]    — all contrast group keys in order
    group_to_color            : dict         — contrast group key → RGB color tuple
    COLOR_MAPS                : dict         — attribute name → {value: hex color}
    INPUT_DEF                 : str          — axis label (used for both rows and columns)
    layer                     : str | None   — "transcriptomics" / "proteomics" / "metabolomics".
                                                When given, loads that layer's
                                                contrast_groups_<layer>.json via
                                                set_contrast_groups() before building.
                                                If None, whatever contrast_groups is
                                                already set (global) is used as-is.
    contrast_groups_base_dir  : str          — base_dir passed through to load_contrast_groups
    """

    if layer is not None:
        set_contrast_groups(layer, base_dir=contrast_groups_base_dir)

    print(f"\nBuilding matrix for {sp} ({len(sp_samples)} samples)...")

    # ── 1. Pearson within-layer correlation matrix ─────────────────────────────
    # Symmetric: cell [s1, s2] = Pearson r between data[s1] and data[s2]
    # across shared non-NaN genes. Diagonal [s,s] = self-correlation = 1.0.
    mat            = pd.DataFrame(np.nan, index=sp_samples, columns=sp_samples)
    n_mat          = pd.DataFrame(0,      index=sp_samples, columns=sp_samples)
    excluded_pairs = []

    for s1 in sp_samples:
        for s2 in sp_samples:
            v1      = data[s1]
            v2      = data[s2]
            mask    = v1.notna() & v2.notna()
            n_valid = mask.sum()
            n_mat.loc[s1, s2] = n_valid
            if n_valid >= 3:
                r, _ = pearsonr(v1[mask], v2[mask])
                mat.loc[s1, s2] = r
            else:
                excluded_pairs.append((s1, s2, int(n_valid)))

    if excluded_pairs:
        print(f"{len(excluded_pairs)} comparison(s) excluded"
              f"(fewer than 3 overlapping genes):")
        for s1, s2, n in excluded_pairs:
            print(f"      {s1}  vs  {s2}  →  {n} gene(s) in common")
    else:
        print(f"  ✓  No comparisons excluded for {sp}.")

    # ── 1b. Colormaps ──────────────────────────────────────────────────────────
    import copy
    cmap_bad   = copy.copy(plt.get_cmap("PuOr"))
    cmap_bad.set_bad(color="black")

    cmap_genes = copy.copy(plt.get_cmap("Blues"))
    cmap_genes.set_bad(color="black")

    # ── 2. Hierarchical clustering (NaN-aware) ─────────────────────────────────
    # Matrix is symmetric so one linkage serves both axes
    lkg     = linkage(nan_aware_condensed_distances(mat.values), method="average")
    row_lkg = lkg
    col_lkg = lkg

    # ── 3. Labels and per-sample colors ───────────────────────────────────────
    original_labels    = mat.index.tolist()
    display_labels     = [make_display_label(s) for s in original_labels]
    label_to_group_map = {lbl: label_to_group(lbl) for lbl in original_labels}

    label_colors = [
        group_to_color.get(label_to_group_map[lbl], "black")
        for lbl in original_labels
    ]

    sp_groups = {
        grp for grp in label_to_group_map.values()
        if grp is not None
    }
    legend_handles = [
        mpatches.Patch(
            facecolor=group_to_color[g],
            label=" | ".join(g.split(" | ")[1:])
        )
        for g in unique_groups
        if g in sp_groups
    ]

    #{sp}_pearson_tree
    # ── 4. Dendrogram ────────────────────────────────────────────────────────── 
    fig, ax = plt.subplots(figsize=(32, 14))

    ddata = dendrogram(
        row_lkg,
        labels=display_labels,
        leaf_rotation=90,
        leaf_font_size=15,
        ax=ax
    )

    leaf_order     = ddata["leaves"]
    ordered_colors = [label_colors[i] for i in leaf_order]
    for tick_label, color in zip(ax.get_xticklabels(), ordered_colors):
        tick_label.set_color(color)

    ax.set_title(
        f"{sp} — Pearson Correlation Tree  ({INPUT_DEF})",
        fontsize=15
    )
    ax.set_xlabel("Samples", fontsize=15)
    ax.set_ylabel("Pearson distance  (1 − r)", fontsize=15)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_tree")
    plt.close()

    # ── 5. Standard clustermap ─────────────────────────────────────────────────
    mat_labeled         = mat.copy()
    mat_labeled.index   = display_labels
    mat_labeled.columns = display_labels

    annot_matrix = mat_labeled.map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )

    clustergrid = sns.clustermap(
        mat_labeled,
        row_linkage=row_lkg,
        col_linkage=col_lkg,
        cmap=cmap_bad,
        center=0,
        figsize=(32, 32),
        linewidths=0.5,
        annot=annot_matrix,
        fmt="",
        annot_kws={"color": "white", "fontsize": 5}
    )

    clustergrid.fig.suptitle(
        f"{sp} — Pearson Clustermap\n{INPUT_DEF}",
        y=1.02,
        fontsize=15
    )
    clustergrid.ax_heatmap.set_xlabel(INPUT_DEF, fontsize=15, labelpad=12)
    clustergrid.ax_heatmap.set_ylabel(INPUT_DEF, fontsize=15, labelpad=12)

    color_clustermap_labels(clustergrid, original_labels, label_to_group_map, group_to_color)

    clustergrid.ax_heatmap.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    save_figure(clustergrid, comparison_folder / f"{safe_name(sp)}_pearson_clustermap")
    plt.close()

    # ── 6. Annotated clustermap ────────────────────────────────────────────────
    build_annotated_heatmap(
        mat_labeled=mat_labeled,
        row_lkg=row_lkg,
        col_lkg=col_lkg,
        original_labels=original_labels,
        display_labels=display_labels,
        sp=sp,
        comparison_folder=comparison_folder,
        unique_groups=unique_groups,
        group_to_color=group_to_color,
        COLOR_MAPS=COLOR_MAPS,
        FIRST_INPUT_DEF=INPUT_DEF,
        SECOND_INPUT_DEF=INPUT_DEF
    )

    # ── 7. Ordered heatmap ─────────────────────────────────────────────────────
    ordered_samples      = sorted(sp_samples, key=_sort_key)
    ordered_display      = [make_display_label(s) for s in ordered_samples]
    ordered_disp_to_orig = dict(zip(ordered_display, ordered_samples))

    mat_ordered         = mat.loc[ordered_samples, ordered_samples].copy()
    mat_ordered.index   = ordered_display
    mat_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(28, 24))

    sns.heatmap(
        mat_ordered,
        ax=ax,
        cmap=cmap_bad,
        center=0,
        linewidths=0.4,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(7)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(7)

    ax.set_title(
        f"{sp} — Pearson r  (numeric / alphabetic order)\n{INPUT_DEF}",
        fontsize=12,
    )
    ax.set_xlabel(INPUT_DEF, fontsize=10, labelpad=12)
    ax.set_ylabel(INPUT_DEF, fontsize=10, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_ordered_heatmap")
    plt.close()

    # ── 8. Gene overlap heatmap ────────────────────────────────────────────────
    # Shows how many genes contributed to each Pearson r. Excluded pairs (< 3)
    # still appear with their real count so you can see exactly why they failed.
    n_labeled         = n_mat.copy()
    n_labeled.index   = display_labels
    n_labeled.columns = display_labels

    n_annot = n_labeled.map(lambda x: str(int(x)) if pd.notna(x) else "")

    # 8a. Clustermap — same row/col ordering as the correlation clustermap
    n_clustergrid = sns.clustermap(
        n_labeled,
        row_linkage=row_lkg,
        col_linkage=col_lkg,
        cmap=cmap_genes,
        figsize=(32, 32),
        linewidths=0.5,
        annot=n_annot,
        fmt="",
        annot_kws={"fontsize": 5}
    )

    n_clustergrid.fig.suptitle(
        f"{sp} — Genes used per correlation\n{INPUT_DEF}",
        y=1.02,
        fontsize=15
    )
    n_clustergrid.ax_heatmap.set_xlabel(INPUT_DEF, fontsize=10, labelpad=12)
    n_clustergrid.ax_heatmap.set_ylabel(INPUT_DEF, fontsize=10, labelpad=12)

    color_clustermap_labels(n_clustergrid, original_labels, label_to_group_map, group_to_color)

    n_clustergrid.ax_heatmap.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )

    save_figure(n_clustergrid, comparison_folder / f"{safe_name(sp)}_gene_overlap_clustermap")
    plt.close()

    # 8b. Ordered heatmap — same numeric/alphabetic ordering as section 7
    n_ordered         = n_mat.loc[ordered_samples, ordered_samples].copy()
    n_ordered.index   = ordered_display
    n_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(28, 24))

    sns.heatmap(
        n_ordered,
        ax=ax,
        cmap=cmap_genes,
        linewidths=0.4,
        linecolor="lightgrey",
        annot=n_ordered.map(lambda x: str(int(x)) if pd.notna(x) else ""),
        fmt="",
        annot_kws={"fontsize": 5},
        cbar_kws={"shrink": 0.6, "label": "genes used"},
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(7)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(7)

    ax.set_title(
        f"{sp} — Genes used per correlation  (numeric / alphabetic order)\n{INPUT_DEF}",
        fontsize=15,
    )
    ax.set_xlabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.set_ylabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_gene_overlap_ordered_heatmap")
    plt.close()

"""
make_correlations_additions.py
────────────────────────────────
Paste the function below into make_correlations.py (e.g. right after
build_and_save_single()). It relies only on names already defined /
imported at the top of that file:

    re, pd, np, plt, mpatches, sns, linkage, dendrogram, squareform,
    pearsonr, Path, copy (imported locally inside build_and_save_single —
    add `import copy` to the top-level imports of make_correlations.py if
    it isn't already there at module level)

    safe_name, make_display_label, label_to_group, build_annotation_matrices,
    nan_aware_condensed_distances, save_figure, _sort_key

Nothing else needs to change in make_correlations.py.
"""


def build_single_layer_tree_and_heatmaps(sp, sp_samples, data, comparison_folder,
                                          unique_groups, group_to_color, COLOR_MAPS,
                                          INPUT_DEF, show_attribute_strips=True,
                                          layer=None, contrast_groups_base_dir=r"C:\Users\papan\Desktop\Coffeeomics"):
    """
    Trimmed single-layer variant of build_and_save_single(): correlates one
    dataset's samples against themselves (symmetric matrix) and saves ONLY:

      1. {sp}_pearson_tree                  - dendrogram, with the
                                               Temperature/CO2/Water-Stress
                                               attribute strips and stacked
                                               legend layout used by the
                                               cross-layer build_and_save()
                                               dendrogram (left orientation,
                                               wide figure, strips beside
                                               the tree).
      2. {sp}_pearson_ordered_heatmap       - Pearson r, samples in
                                               numeric/alphabetic order.
      3. {sp}_gene_overlap_ordered_heatmap  - feature-count-per-pair,
                                               same ordering.

    The plain clustermap, annotated clustermap, and gene-overlap clustermap
    from build_and_save_single() are intentionally skipped.

    Parameters
    ----------
    sp                    : str          — species name
    sp_samples            : list[str]    — sample column names belonging to this species
    data                  : pd.DataFrame — single omics layer (features x samples)
    comparison_folder     : Path         — output folder for saved figures
    unique_groups         : list[str]    — all contrast group keys in order
    group_to_color        : dict         — contrast group key -> RGB color tuple
    COLOR_MAPS            : dict         — attribute name -> {value: hex color}
    INPUT_DEF             : str          — axis label (used for both rows and columns)
    show_attribute_strips : bool         — draw Temperature/CO2/Water-Stress
                                            strips beside the dendrogram (default True)
    layer                  : str | None  — "transcriptomics" / "proteomics" / "metabolomics".
                                            When given, loads that layer's
                                            contrast_groups_<layer>.json via
                                            set_contrast_groups() before building.
                                            If None, whatever contrast_groups is
                                            already set (global) is used as-is.
    contrast_groups_base_dir : str       — base_dir passed through to load_contrast_groups
    """

    if layer is not None:
        set_contrast_groups(layer, base_dir=contrast_groups_base_dir)

    print(f"\nBuilding matrix for {sp} ({len(sp_samples)} samples)...")

    # ── 1. Pearson within-layer correlation matrix (symmetric) ────────────
    mat            = pd.DataFrame(np.nan, index=sp_samples, columns=sp_samples)
    n_mat          = pd.DataFrame(0,      index=sp_samples, columns=sp_samples)
    excluded_pairs = []

    for s1 in sp_samples:
        for s2 in sp_samples:
            v1      = data[s1]
            v2      = data[s2]
            mask    = v1.notna() & v2.notna()
            n_valid = mask.sum()
            n_mat.loc[s1, s2] = n_valid
            if n_valid >= 3:
                r, _ = pearsonr(v1[mask], v2[mask])
                mat.loc[s1, s2] = r
            else:
                excluded_pairs.append((s1, s2, int(n_valid)))

    if excluded_pairs:
        print(f"  ⚠  {len(excluded_pairs)} comparison(s) excluded "
              f"(fewer than 3 overlapping features):")
        for s1, s2, n in excluded_pairs:
            print(f"      {s1}  vs  {s2}  →  {n} feature(s) in common")
    else:
        print(f"  ✓  No comparisons excluded for {sp}.")

    # ── 1b. Colormaps ──────────────────────────────────────────────────────
    import copy
    cmap_bad   = copy.copy(plt.get_cmap("PuOr"))
    cmap_bad.set_bad(color="black")

    cmap_genes = copy.copy(plt.get_cmap("Blues"))
    cmap_genes.set_bad(color="black")

    # ── 2. Hierarchical clustering (NaN-aware) — symmetric, one linkage ───
    lkg     = linkage(nan_aware_condensed_distances(mat.values), method="average")
    row_lkg = lkg

    # ── 3. Labels and per-sample colors ────────────────────────────────────
    original_labels    = mat.index.tolist()
    display_labels     = [make_display_label(s) for s in original_labels]
    label_to_group_map = {lbl: label_to_group(lbl) for lbl in original_labels}

    label_colors = [
        group_to_color.get(label_to_group_map[lbl], "black")
        for lbl in original_labels
    ]

    sp_groups = {
        grp for grp in label_to_group_map.values()
        if grp is not None
    }
    legend_handles = [
        mpatches.Patch(
            facecolor=group_to_color[g],
            label=" | ".join(g.split(" | ")[1:])
        )
        for g in unique_groups
        if g in sp_groups
    ]

    # ── 4. Dendrogram — cross-layer aesthetic, WITH attribute strips ───────
    attributes = ["Temperature", "CO2_ppm", "Water_Stress"]

    if show_attribute_strips:
        annotation_matrices, _ = build_annotation_matrices(display_labels, original_labels)
        n_attrs = len(attributes)
        fig, axes = plt.subplots(
            1, 2 + n_attrs,   # tree, spacer, then one axis per strip
            figsize=(14 + 0.8 * n_attrs, 32),
            gridspec_kw={
                "width_ratios": [10, 8] + [0.35] * n_attrs,
                "wspace": 0.05,
            },
        )
        ax = axes[0]
        spacer_ax = axes[1]
        spacer_ax.axis("off")
        strip_axes = list(axes[2:])
    else:
        fig, ax = plt.subplots(figsize=(8, 32))
        strip_axes = []

    ddata = dendrogram(
        row_lkg,
        labels=display_labels,
        orientation="left",
        leaf_rotation=0,
        leaf_font_size=15,
        ax=ax
    )

    leaf_order     = ddata["leaves"]
    ordered_colors = [label_colors[i] for i in leaf_order]
    for tick_label, color in zip(ax.get_yticklabels(), ordered_colors):
        tick_label.set_color(color)

    if show_attribute_strips:
        y_positions = ax.get_yticks()

        for strip_ax, attr in zip(strip_axes, attributes):
            colors_ordered = [annotation_matrices[attr].iloc[i, 0] for i in leaf_order]
            for y, color in zip(y_positions, colors_ordered):
                strip_ax.add_patch(
                    plt.Rectangle((0, y - 5), 1, 10, facecolor=color, edgecolor="white")
                )
            strip_ax.set_ylim(ax.get_ylim())
            strip_ax.set_xlim(0, 1)
            strip_ax.set_xticks([])
            strip_ax.set_yticks([])
            strip_ax.set_title(attr.replace("_", " "), fontsize=15, rotation=90, va="bottom")
            for spine in strip_ax.spines.values():
                spine.set_visible(False)

    ax.set_title(
        f"{sp} — Pearson Correlation Tree\n{INPUT_DEF}",
        fontsize=15
    )
    ax.set_ylabel("Samples", fontsize=15)
    ax.set_xlabel("Pearson distance  (1 − r)", fontsize=15)

    leg_group = fig.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(-0.2, 0.8),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True
    )
    fig.add_artist(leg_group)

    if show_attribute_strips:
        for i, attr in enumerate(attributes):
            handles = [
                mpatches.Patch(facecolor=color, label=val)
                for val, color in COLOR_MAPS[attr].items()
            ]
            leg = fig.legend(
                handles=handles,
                title=attr.replace("_", " "),
                loc="upper left",
                bbox_to_anchor=(-0.2, 0.6 - i * 0.08),
                fontsize=15,
                title_fontsize=15,
                frameon=True
            )
            fig.add_artist(leg)

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_tree")
    plt.close()

    # ── 5. Ordered heatmap (Pearson r) ──────────────────────────────────────
    ordered_samples      = sorted(sp_samples, key=_sort_key)
    ordered_display      = [make_display_label(s) for s in ordered_samples]
    ordered_disp_to_orig = dict(zip(ordered_display, ordered_samples))

    mat_ordered         = mat.loc[ordered_samples, ordered_samples].copy()
    mat_ordered.index   = ordered_display
    mat_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(30, 24))

    sns.heatmap(
        mat_ordered,
        ax=ax,
        cmap=cmap_bad,
        center=0,
        linewidths=0.4,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(15)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(15)

    ax.set_title(
        f"{sp} — Pearson r  (numeric / alphabetic order)\n{INPUT_DEF}",
        fontsize=15,
    )
    ax.set_xlabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.set_ylabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_pearson_ordered_heatmap")
    plt.close()

    # ── 6. Gene/feature overlap ordered heatmap ─────────────────────────────
    n_ordered         = n_mat.loc[ordered_samples, ordered_samples].copy()
    n_ordered.index   = ordered_display
    n_ordered.columns = ordered_display

    fig, ax = plt.subplots(figsize=(28, 24))

    sns.heatmap(
        n_ordered,
        ax=ax,
        cmap=cmap_genes,
        linewidths=0.4,
        linecolor="lightgrey",
        annot=n_ordered.map(lambda x: str(int(x)) if pd.notna(x) else ""),
        fmt="",
        annot_kws={"fontsize": 5},
        cbar_kws={"shrink": 0.6, "label": "features used"},
    )

    for tick in ax.get_xticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(90)
        tick.set_fontsize(15)

    for tick in ax.get_yticklabels():
        orig = ordered_disp_to_orig.get(tick.get_text(), tick.get_text())
        grp  = label_to_group_map.get(orig)
        tick.set_color(group_to_color.get(grp, "black"))
        tick.set_rotation(0)
        tick.set_fontsize(15)

    ax.set_title(
        f"{sp} — Features used per correlation  (numeric / alphabetic order)\n{INPUT_DEF}",
        fontsize=15,
    )
    ax.set_xlabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.set_ylabel(INPUT_DEF, fontsize=15, labelpad=12)
    ax.legend(
        handles=legend_handles,
        title="Contrast group",
        bbox_to_anchor=(1.25, 1),
        loc="upper left",
        fontsize=15,
        title_fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    save_figure(fig, comparison_folder / f"{safe_name(sp)}_gene_overlap_ordered_heatmap")
    plt.close()