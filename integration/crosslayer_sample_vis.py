"""
plot_sample_overview_with_ids.py
---------------------------------
Recreates MOFA2's plot_data_overview() (transcriptomics / proteomics /
metabolomics presence per species group), but with actual sample IDs
labeled on the x-axis instead of anonymous tiles -- useful for identifying
exactly which samples are driving a missingness pattern (e.g. the
contiguous transcriptomics/proteomics gap vs. the scattered metabolomics
gaps) without cross-referencing separately.

Reads directly from the already-corrected long-format MOFA input
(mofa_input_long.csv, produced by build_mofa_input.py), so this reflects
the actual corrected data that went into the model, not the raw study
design or uncorrected contrast_groups files.

>>> EDIT THE "CONFIG" SECTION BELOW BEFORE RUNNING <<<
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================================
# CONFIG -- edit these paths and styling for your machine
# =========================================================================

BASE_DIR = Path(r"C:\Users\papan\Desktop\Coffeeomics")

# Output of build_mofa_input.py
LONG_INPUT_PATH = BASE_DIR / "integration" / "mofa2" / "mofa_input_long.csv"

OUTPUT_FIGURE_PATH = BASE_DIR / "integration" / "sample_overview_with_ids.png"

VIEW_ORDER = ["transcriptomics", "proteomics", "metabolomics"]
VIEW_COLORS = {
    "transcriptomics": "#3541AF",
    "proteomics": "#FF8331",
    "metabolomics": "#89D921",
}
MISSING_COLOR = "#656565"  # tile color for "no data in this view"

FIGSIZE_PER_SAMPLE = 0.16  # inches of figure width per sample column -- increase if labels overlap
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 11
DPI = 200

# =========================================================================
# Sample ID sort key: numeric-then-letter, not plain string sort
# =========================================================================

_SAMPLE_ID_RE = re.compile(r"^(\d+)([A-Za-z]*)$")


def sample_sort_key(sample_id: str):
    """
    Sorts "7A" < "7B" < "9A" < "10A" -- a plain string sort would
    incorrectly put "10A" before "7A" and "9A" between them.
    """
    m = _SAMPLE_ID_RE.match(str(sample_id))
    if m:
        number = int(m.group(1))
        letter = m.group(2)
        return (0, number, letter)
    return (1, 0, str(sample_id))  # unrecognized IDs sort after all numeric ones


# =========================================================================
# Load data
# =========================================================================


def load_long_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_mofa_input.py first to generate it."
        )
    df = pd.read_csv(path)
    required = {"sample", "group", "feature", "view", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing column(s): {missing}")
    return df


def get_feature_counts(df: pd.DataFrame) -> dict:
    """Number of unique features per view, for the row labels (D=...)."""
    return df.groupby("view")["feature"].nunique().to_dict()


# =========================================================================
# Build presence matrix per group
# =========================================================================


def build_presence_matrix(df: pd.DataFrame, group: str):
    """
    Returns (ordered_sample_ids, {view_name: set_of_samples_present}) for
    one group (species) -- ordered_sample_ids is the union of samples
    across all views for that group, sorted with sample_sort_key.
    """
    group_df = df[df["group"] == group]

    presence = {}
    all_samples = set()
    for view in VIEW_ORDER:
        view_samples = set(group_df.loc[group_df["view"] == view, "sample"].unique())
        presence[view] = view_samples
        all_samples |= view_samples

    ordered_samples = sorted(all_samples, key=sample_sort_key)
    return ordered_samples, presence


# =========================================================================
# Plotting
# =========================================================================


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def plot_group_panel(ax, group: str, ordered_samples: list, presence: dict,
                      feature_counts: dict) -> None:
    n_samples = len(ordered_samples)
    n_views = len(VIEW_ORDER)

    img = np.zeros((n_views, n_samples, 3))
    for row_idx, view in enumerate(VIEW_ORDER):
        present_rgb = _hex_to_rgb(VIEW_COLORS[view])
        missing_rgb = _hex_to_rgb(MISSING_COLOR)
        for col_idx, sample in enumerate(ordered_samples):
            img[row_idx, col_idx] = present_rgb if sample in presence[view] else missing_rgb

    ax.imshow(img, aspect="auto", interpolation="none")

    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(ordered_samples, rotation=90, fontsize=LABEL_FONTSIZE)

    ax.set_yticks(range(n_views))
    y_labels = [f"{view}\nD={feature_counts.get(view, '?')}" for view in VIEW_ORDER]
    ax.set_yticklabels(y_labels, fontsize=9)

    ax.set_title(f"{group}\nN={n_samples}", fontsize=TITLE_FONTSIZE)

    ax.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_views, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# =========================================================================
# Main
# =========================================================================


def main():
    df = load_long_input(LONG_INPUT_PATH)
    feature_counts = get_feature_counts(df)
    groups = sorted(df["group"].unique())

    group_data = {}
    max_samples = 0
    for group in groups:
        ordered_samples, presence = build_presence_matrix(df, group)
        group_data[group] = (ordered_samples, presence)
        max_samples = max(max_samples, len(ordered_samples))
        print(f"{group}: N={len(ordered_samples)} unique samples across all views")

    fig_width = max(10, max_samples * FIGSIZE_PER_SAMPLE)
    fig_height = 2.4 * len(groups)

    fig, axes = plt.subplots(len(groups), 1, figsize=(fig_width, fig_height))
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        ordered_samples, presence = group_data[group]
        plot_group_panel(ax, group, ordered_samples, presence, feature_counts)

    plt.tight_layout()
    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved figure to {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()