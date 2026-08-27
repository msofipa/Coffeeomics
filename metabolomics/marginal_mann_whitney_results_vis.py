"""
Build UpSet plots for the marginal Mann-Whitney U metabolomics results.

For every "*_marginal.xlsx" file in the marginal results folder (e.g.
co2_mannwhitneyu_marginal.xlsx, temperature_mannwhitneyu_marginal.xlsx,
water_stress_mannwhitneyu_marginal.xlsx, ...), and for each species sheet
inside it, this produces ONE UpSet plot where:

    - each SET is one condition/contrast column in that file (the marginal
      pairwise comparison for that stressor), and
    - the plot shows how significantly differentially abundant metabolites
      (DAMs, green-highlighted cells with 0 < p <= 0.05) overlap across
      those contrasts.

Reuses the same header-reconstruction / green-cell-detection logic as
count_green_highlights.ipynb (openpyxl, not pandas.read_excel, since we need
actual fill colors and the merged multi-level header has to be forward-filled
to recover real contrast labels).

Requires: pip install openpyxl pandas matplotlib upsetplot

NOTE on upsetplot 0.9.0 + numpy>=2.0 / pandas>=2 compatibility:
upsetplot 0.9.0 has two latent bugs that only surface with numpy>=2.0
(and, for one of them, pandas>=2 running with copy-on-write, which is the
default in pandas>=3.0):
  1. UpSet.plot_matrix fills missing dot-styling values via
     `styles[col].fillna(x, inplace=True)`, which silently no-ops under
     pandas copy-on-write, leaving NaN styling values that crash
     matplotlib's ax.scatter with "Invalid RGBA argument: nan".
  2. UpSet._label_sizes computes a text-position margin as
     `0.01 * abs(np.diff(ax.get_xlim()))`, a size-1 numpy array rather
     than a scalar; numpy>=2.0 no longer allows implicitly converting
     that to a Python float, raising "only 0-dimensional arrays can be
     converted to Python scalars".
Rather than requiring you to downgrade numpy/matplotlib in your venv
(which can require a C compiler on Windows to build from source), this
script monkeypatches both methods with corrected versions at import time.
The patch is self-contained below and only touches upsetplot's in-memory
class, not the installed package -- safe to remove if a future upsetplot
release fixes this upstream.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from upsetplot import UpSet, from_contents, util
from upsetplot.plotting import UpSet as _UpSetClass
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Monkeypatch: fix upsetplot 0.9.0's numpy>=2.0 / pandas copy-on-write bugs
# (see module docstring for details). Both replacements are otherwise
# identical to the upstream implementations in upsetplot/plotting.py.
# ----------------------------------------------------------------------

def _patched_plot_matrix(self, ax):
    ax = self._reorient(ax)
    data = self.intersections
    n_cats = data.index.nlevels

    inclusion = data.index.to_frame().values

    styles = [
        [
            self.subset_styles[i]
            if inclusion[i, j]
            else {"facecolor": self._other_dots_color, "linewidth": 0}
            for j in range(n_cats)
        ]
        for i in range(len(data))
    ]
    styles = sum(styles, [])
    style_columns = {
        "facecolor": "facecolors",
        "edgecolor": "edgecolors",
        "linewidth": "linewidths",
        "linestyle": "linestyles",
        "hatch": "hatch",
    }
    styles = (
        pd.DataFrame(styles)
        .reindex(columns=style_columns.keys())
        .astype(
            {
                "facecolor": "O",
                "edgecolor": "O",
                "linewidth": float,
                "linestyle": "O",
                "hatch": "O",
            }
        )
    )
    # FIX: non-inplace fillna (inplace silently no-ops under pandas CoW)
    styles["linewidth"] = styles["linewidth"].fillna(1)
    styles["facecolor"] = styles["facecolor"].fillna(self._facecolor)
    styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"])
    styles["linestyle"] = styles["linestyle"].fillna("solid")
    del styles["hatch"]

    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))

    if self._element_size is not None:
        s = (self._element_size * 0.35) ** 2
    else:
        s = 200
    ax.scatter(
        *self._swapaxes(x, y),
        s=s,
        zorder=10,
        **styles.rename(columns=style_columns),
    )

    if self._with_lines:
        idx = np.flatnonzero(inclusion)
        line_data = (
            pd.Series(y[idx], index=x[idx])
            .groupby(level=0)
            .aggregate(["min", "max"])
        )
        colors = pd.Series(
            [
                style.get("edgecolor", style.get("facecolor", self._facecolor))
                for style in self.subset_styles
            ],
            name="color",
        )
        line_data = line_data.join(colors)
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=2,
            colors=line_data["color"],
            zorder=5,
        )

    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(
        data.index.names, rotation=0 if self._horizontal else -90
    )
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not self._horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
    ax.grid(False)


def _patched_label_sizes(self, ax, rects, where):
    if not self._show_counts and not self._show_percentages:
        return
    if self._show_counts is True:
        count_fmt = "{:.0f}"
    else:
        count_fmt = self._show_counts
        if "{" not in count_fmt:
            count_fmt = util.to_new_pos_format(count_fmt)

    pct_fmt = "{:.1%}" if self._show_percentages is True else self._show_percentages

    if count_fmt and pct_fmt:
        if where == "top":
            fmt = f"{count_fmt}\n({pct_fmt})"
        else:
            fmt = f"{count_fmt} ({pct_fmt})"

        def make_args(val):
            return val, val / self.total
    elif count_fmt:
        fmt = count_fmt

        def make_args(val):
            return (val,)
    else:
        fmt = pct_fmt

        def make_args(val):
            return (val / self.total,)

    if where == "right":
        # FIX: unwrap size-1 array to a plain float (numpy>=2.0 no longer
        # implicitly converts non-0-d arrays to scalars)
        margin = float(0.01 * abs(np.diff(ax.get_xlim()))[0])
        for rect in rects:
            width = rect.get_width() + rect.get_x()
            ax.text(
                width + margin,
                rect.get_y() + rect.get_height() * 0.5,
                fmt.format(*make_args(width)),
                ha="left",
                va="center",
            )
    elif where == "left":
        margin = float(0.01 * abs(np.diff(ax.get_xlim()))[0])
        for rect in rects:
            width = rect.get_width() + rect.get_x()
            ax.text(
                width + margin,
                rect.get_y() + rect.get_height() * 0.5,
                fmt.format(*make_args(width)),
                ha="right",
                va="center",
            )
    elif where == "top":
        margin = float(0.01 * abs(np.diff(ax.get_ylim()))[0])
        for rect in rects:
            height = rect.get_height() + rect.get_y()
            ax.text(
                rect.get_x() + rect.get_width() * 0.5,
                height + margin,
                fmt.format(*make_args(height)),
                ha="center",
                va="bottom",
            )
    else:
        raise NotImplementedError("unhandled where: %r" % where)


_UpSetClass.plot_matrix = _patched_plot_matrix
_UpSetClass._label_sizes = _patched_label_sizes

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Users\papan\Desktop\Coffeeomics")
MARGINAL_FOLDER = PROJECT_ROOT / "metabolomics" / "mann_whitney_results" / "marginal"
FILE_GLOB = "*_marginal.xlsx"
SHEET_NAMES = ["C. arabica", "C. canephora"]
GREEN_HEX = "32A852"  # matches "background-color: #32a852"

OUTPUT_FOLDER = MARGINAL_FOLDER / "upset_plots"

# Colorblind-safe qualitative palette (Okabe & Ito, 2008 -- distinguishable
# under the common forms of color vision deficiency: deuteranopia,
# protanopia, tritanopia). Cycled through per-set, in a fixed order
# (alphabetical by contrast label) so colors are stable across reruns.
COLORBLIND_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# If a file/species has more contrasts than this, only the TOP_N contrasts
# with the most DAMs are kept as sets (UpSet plots become unreadable well
# past ~8-10 sets). Set to None to always keep every contrast.
TOP_N_CONTRASTS = 10

# Drop contrasts with zero DAMs (empty sets add nothing to an UpSet plot
# and upsetplot errors on all-empty input).
DROP_EMPTY_CONTRASTS = True

# Recovery-condition ("Rec 14 - 25/20°C") samples/contrasts must be excluded
# from the temperature-collapsed (marginal) analysis. Any contrast whose
# label contains one of these substrings is dropped before plotting, for
# any file whose stressor name matches a key here (checked by substring
# against the file's stem, e.g. "temperature" matches
# "temperature_mannwhitneyu_marginal.xlsx"). Add more stressor: [markers]
# entries if other files need similar exclusions.
EXCLUDED_CONTRAST_MARKERS_BY_STRESSOR = {
    "temperature": ["Rec 14"],
}


# ----------------------------------------------------------------------
# Green-cell detection (same convention as count_green_highlights.ipynb)
# ----------------------------------------------------------------------

def is_green(cell) -> bool:
    """True if this cell's fill matches GREEN_HEX (solid fill)."""
    fill = cell.fill
    if fill is None or fill.patternType != "solid":
        return False

    color = fill.fgColor
    if color is None or color.type != "rgb" or not color.rgb:
        return False

    return color.rgb[-6:].upper() == GREEN_HEX.upper()


# ----------------------------------------------------------------------
# Multi-level header reconstruction (forward-fill across merged cells)
# ----------------------------------------------------------------------

def get_header_and_labels(ws: Worksheet):
    """
    Returns:
    - stat_row: 1-indexed row number containing the 'u_statistic'/'p_value'
      labels (the innermost, bottom-most header row)
    - col_labels: {col_idx: (level0, level1, ..., 'u_statistic'|'p_value')},
      forward-filled across merged header cells
    """
    max_col = ws.max_column
    max_row = ws.max_row

    stat_row = None
    for r in range(1, max_row + 1):
        row_values = [ws.cell(row=r, column=c).value for c in range(1, max_col + 1)]
        if "p_value" in row_values or "u_statistic" in row_values:
            stat_row = r
            break

    if stat_row is None:
        raise ValueError("Could not find header row containing 'u_statistic'/'p_value'")

    header_rows = []
    for r in range(1, stat_row + 1):
        values = []
        last = None
        for c in range(1, max_col + 1):
            v = ws.cell(row=r, column=c).value
            if v is not None:
                last = v
            values.append(last)
        header_rows.append(values)

    col_labels = {}
    for c in range(1, max_col + 1):
        levels = tuple(header_rows[lvl][c - 1] for lvl in range(stat_row))
        col_labels[c] = levels

    return stat_row, col_labels


def extract_highlighted_records(ws: Worksheet, file_name: str, species: str):
    """
    Returns a list of dicts, one per green cell:
    {file, species, metabolite, contrast}
    'metabolite' is the real row label (column A); 'contrast' is the real
    reconstructed column label (all header levels except the trailing
    'p_value' marker, joined with " | ").
    """
    stat_row, col_labels = get_header_and_labels(ws)
    records = []

    for row in ws.iter_rows(min_row=stat_row + 1):
        for cell in row:
            if cell.value is None or not is_green(cell):
                continue

            levels = col_labels.get(cell.column, ())
            if not levels or levels[-1] != "p_value":
                # highlighting was only ever applied to p_value columns;
                # skip anything else as a safety net
                continue

            metabolite = ws.cell(row=cell.row, column=1).value
            contrast = " | ".join(str(x) for x in levels[:-1])

            records.append(
                {
                    "file": file_name,
                    "species": species,
                    "metabolite": metabolite,
                    "contrast": contrast,
                }
            )

    return records


# ----------------------------------------------------------------------
# Colorblind-safe per-set coloring
# ----------------------------------------------------------------------

def assign_set_colors(labels):
    """
    Returns {label: hex color}, one color per label, drawn from
    COLORBLIND_PALETTE in a fixed (alphabetical) order so the same
    contrast always gets the same color across separate runs. If there
    are more labels than palette colors, colors repeat (a legend/label
    is still needed to disambiguate beyond 8 sets, which is already the
    practical limit for a readable UpSet plot).
    """
    return {
        label: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, label in enumerate(sorted(labels))
    }


# ----------------------------------------------------------------------
# UpSet plot construction
# ----------------------------------------------------------------------

def build_upset_sets(records_df: pd.DataFrame, top_n: int | None, stressor: str):
    """
    records_df: rows for a single (file, species), columns metabolite/contrast.
    Returns {contrast_label: set(metabolite names)}, optionally restricted to
    the top_n contrasts by DAM count, and with any contrast matching
    EXCLUDED_CONTRAST_MARKERS_BY_STRESSOR[stressor] removed first (e.g.
    Rec 14 recovery-condition contrasts are excluded from temperature).
    """
    excluded_markers = []
    for keyword, markers in EXCLUDED_CONTRAST_MARKERS_BY_STRESSOR.items():
        if keyword in stressor.lower():
            excluded_markers.extend(markers)

    if excluded_markers:
        before = records_df["contrast"].nunique()
        mask = ~records_df["contrast"].apply(
            lambda c: any(marker in c for marker in excluded_markers)
        )
        excluded_labels = sorted(set(records_df.loc[~mask, "contrast"]))
        records_df = records_df[mask]
        if excluded_labels:
            print(
                f"  Excluding {len(excluded_labels)} contrast(s) for "
                f"stressor {stressor!r} matching {excluded_markers}: "
                f"{excluded_labels}"
            )

    contrast_to_metabolites = (
        records_df.groupby("contrast")["metabolite"].apply(lambda s: set(s.dropna()))
    )

    if DROP_EMPTY_CONTRASTS:
        contrast_to_metabolites = contrast_to_metabolites[
            contrast_to_metabolites.map(len) > 0
        ]

    if top_n is not None and len(contrast_to_metabolites) > top_n:
        contrast_to_metabolites = contrast_to_metabolites.loc[
            contrast_to_metabolites.map(len).sort_values(ascending=False).index[:top_n]
        ]

    return contrast_to_metabolites.to_dict()


def plot_upset(sets_dict: dict, title: str, out_path: Path):
    if len(sets_dict) < 2:
        print(
            f"  Skipped '{title}': need >= 2 non-empty contrasts for an UpSet "
            f"plot, found {len(sets_dict)}."
        )
        return

    contents = from_contents(sets_dict)
    upset = UpSet(
        contents,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    )
    fig = plt.figure(figsize=(max(8, 1.2 * len(sets_dict)), 6))
    axes = upset.plot(fig=fig)

    # Recolor each set's marginal-total bar with a distinct colorblind-safe
    # color, purely to make the bars visually distinguishable at a glance --
    # colors are NOT tied to condition/attribute meaning. The totals axis'
    # yticklabels give the definitive contrast-to-patch order (upsetplot's
    # internal sort order for the intersection matrix does not reorder
    # this axis).
    totals_ax = axes.get("totals")
    if totals_ax is not None:
        labels = [t.get_text() for t in totals_ax.get_yticklabels()]
        set_colors = assign_set_colors(labels)
        for label, patch in zip(labels, totals_ax.patches):
            patch.set_facecolor(set_colors[label])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def export_intersections_csv(sets_dict: dict, out_path: Path):
    """
    Writes one row per exact-membership intersection (matching the UpSet
    plot's bar semantics: a metabolite is counted in exactly one row, the
    one listing every contrast it is significant in -- not "at least
    these"). Columns: combination, n_contrasts, n_metabolites, metabolites
    (semicolon-separated, alphabetical). Sorted the same way the UpSet
    plot bars are (largest intersection first).
    """
    if not sets_dict:
        print(f"  Skipped CSV for {out_path.name}: no non-empty contrasts.")
        return

    contents = from_contents(sets_dict).reset_index()
    contrast_cols = [c for c in contents.columns if c != "id"]

    def combination_label(row):
        included = [c for c in contrast_cols if row[c]]
        return " & ".join(included)

    contents["combination"] = contents.apply(combination_label, axis=1)

    grouped = (
        contents.groupby("combination")["id"]
        .apply(lambda s: sorted(s))
        .reset_index()
        .rename(columns={"id": "metabolites"})
    )
    grouped["n_contrasts"] = grouped["combination"].str.count(r" & ") + 1
    grouped["n_metabolites"] = grouped["metabolites"].map(len)
    grouped = grouped.sort_values(
        ["n_metabolites", "n_contrasts"], ascending=[False, False]
    ).reset_index(drop=True)
    grouped["metabolites"] = grouped["metabolites"].map("; ".join)
    grouped = grouped[["combination", "n_contrasts", "n_metabolites", "metabolites"]]

    # utf-8-sig so Excel on Windows correctly displays the ° characters
    # in temperature contrast labels rather than mojibake.
    grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(
        p for p in MARGINAL_FOLDER.glob(FILE_GLOB) if not p.name.startswith("~$")
    )
    if not xlsx_files:
        print(f"No files matching {FILE_GLOB!r} found in: {MARGINAL_FOLDER}")
        return

    for path in xlsx_files:
        print(f"Reading: {path.name}")
        wb = load_workbook(path, data_only=True)

        all_records = []
        for sheet_name in SHEET_NAMES:
            if sheet_name not in wb.sheetnames:
                print(
                    f"  Warning: sheet {sheet_name!r} not found in {path.name} "
                    f"(available: {wb.sheetnames}) -- skipped"
                )
                continue
            ws = wb[sheet_name]
            all_records.extend(extract_highlighted_records(ws, path.name, sheet_name))

        wb.close()

        records_df = pd.DataFrame(all_records)
        if records_df.empty:
            print(f"  No green-highlighted (significant) cells found in {path.name}")
            continue

        stressor = path.stem.replace("_mannwhitneyu_marginal", "").replace(
            "_marginal", ""
        )

        for sheet_name in SHEET_NAMES:
            species_df = records_df[records_df["species"] == sheet_name]
            if species_df.empty:
                continue

            sets_dict = build_upset_sets(species_df, TOP_N_CONTRASTS, stressor)
            species_slug = sheet_name.replace(". ", "_").replace(" ", "_")

            out_path = OUTPUT_FOLDER / f"upset_{stressor}_{species_slug}.png"
            title = f"{stressor} | {sheet_name}: DAM overlap across contrasts"
            plot_upset(sets_dict, title, out_path)

            csv_path = OUTPUT_FOLDER / f"intersections_{stressor}_{species_slug}.csv"
            export_intersections_csv(sets_dict, csv_path)


if __name__ == "__main__":
    main()