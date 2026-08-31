"""Shared Nature-journal-style rcParams and validated colorblind-safe
palette for both figure-generation modules (figures.py, landmark_figures.py).

Palette values come from the project's `dataviz` skill's validated default
palette -- computed and checked for CVD-safety (Machado-Oliveira-Fernandes
2009 simulation), not eyeballed. Categorical hues are assigned in this fixed
order to a small set of genuine *identities* (model variant, threshold,
quarter) and never reordered per-chart or reused for a magnitude/count axis
(e.g. 30 teams, 5 seasons) -- those get one sequential hue instead.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed order, never cycled: identity 1 always reads as CAT_COLORS[0], etc.
CAT_COLORS = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential (magnitude, light -> dark). For an ordinal ramp (discrete
# ordered marks) the lightest usable step is 250, not 100 -- see the
# dataviz skill's palette reference.
SEQ_BLUE = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}

# Diverging (polarity): blue <-> neutral gray <-> red, equal steps per arm.
# Not seaborn/matplotlib's 'coolwarm' -- a distinct, validated pair.
DIVERGING = LinearSegmentedColormap.from_list(
    "choke_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)

# A separate, reserved scale -- never reused for series/variant identity.
STATUS_COLORS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Ink hierarchy. Text wears text tokens, never a series color -- a colored mark
# beside a label carries the identity, the label itself stays neutral. Three
# levels so a reader's eye lands on titles first and tick numbers last.
INK = {
    "primary": "#1a1a1a",  # titles, panel letters
    "secondary": "#3d3d3d",  # axis labels, legend text
    "muted": "#6e6e6e",  # tick labels, footnotes, reference lines
}

# Chrome: one shade off the surface, solid hairlines (a dashed grid reads as a
# threshold or projection when it is only a grid -- see the dataviz skill's
# anti-pattern list).
SURFACE = "#ffffff"
GRID_COLOR = "#e6e6e6"
AXIS_COLOR = "#bdbdbd"


def apply_style():
    """Nature-journal rcParams: sans-serif, minimal chrome, no gridlines by
    default, consistent type scale. Call as the FIRST line of any
    generate_all_*_figures() entrypoint -- in particular, before any
    sns.set_theme() call, which would otherwise silently override this.
    """
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.6,
            "axes.edgecolor": AXIS_COLOR,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK["primary"],
            "axes.titlepad": 8,
            "axes.labelsize": 9,
            "axes.labelcolor": INK["secondary"],
            "axes.labelpad": 5,
            "text.color": INK["primary"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.labelcolor": INK["muted"],
            "ytick.labelcolor": INK["muted"],
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.labelcolor": INK["secondary"],
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.4,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "lines.linewidth": 1.4,
            "lines.markersize": 5,
            "patch.linewidth": 0,
            "errorbar.capsize": 0,
            "savefig.dpi": 300,
            "figure.dpi": 100,
        }
    )


def panel_label(ax, letter, **kwargs):
    """Bold panel letter (a, b, c...) just outside the axes' upper-left
    corner -- the Nature multi-panel-figure convention. Returns the artist
    so callers/tests can inspect it.

    The offset is in *points*, not axes fractions: an axes-fraction offset
    scales with the panel, so the same (-0.12, 1.08) that sits snugly beside
    a tall narrow panel drifts far off a wide short one. A fixed point
    offset puts every letter the same physical distance from its corner,
    which is what makes a multi-panel sheet read as one grid.
    """
    defaults = dict(
        fontsize=11, fontweight="bold", color=INK["primary"], va="baseline", ha="left"
    )
    defaults.update(kwargs)
    return ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-26, 8),
        textcoords="offset points",
        **defaults,
    )


def recessive_grid(ax, axis="y"):
    """Turn on a hairline grid one shade off the surface, behind the data.

    Global `axes.grid` stays False (a scatter or forest plot is cleaner
    without one); charts whose job is magnitude comparison opt in here.
    """
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.6, linestyle="-")
    return ax


def row_bands(ax, n_rows, min_rows=5):
    """Very light alternating bands, one per row, on a horizontal chart
    (forest, dot plot, dumbbell). On a 30-row forest the eye has to travel a
    long way from a y-tick label to its mark; banding keeps the row together
    without adding a gridline per row.

    Below `min_rows` the bands stop helping and start reading as a filled
    region: on a two-row panel a single band covers half of it. Callers whose
    panels are wide and short can lower the threshold -- a band there still
    reads as a stripe -- but they must say so explicitly.
    """
    if n_rows < min_rows:
        return
    for i in range(0, n_rows, 2):
        ax.axhspan(
            i - 0.5, i + 0.5, color=GRID_COLOR, alpha=0.45, zorder=0, linewidth=0
        )


def zero_rule(ax):
    """A vertical rule at x=0, darker than the grid. On a forest or a signed
    difference chart, whether a mark or interval crosses zero is a real
    reference the reader is looking for -- not chrome.
    """
    return ax.axvline(0, color=INK["muted"], linewidth=0.8, zorder=1)


def label_values(ax, positions, values, fmt="{:.0f}", orient="v", pad=3, **kwargs):
    """Direct-label a *selected* set of marks (never every point -- a number
    on every mark is chaos and goes unread). Labels wear the muted ink
    token, not the mark's color.

    `orient='v'` places labels above the mark (bars, columns); `'h'` places
    them to the right (horizontal bars, dot plots).
    """
    defaults = dict(fontsize=7, color=INK["muted"])
    defaults.update(kwargs)
    offset = (0, pad) if orient == "v" else (pad, 0)
    align = (
        dict(ha="center", va="bottom")
        if orient == "v"
        else dict(ha="left", va="center")
    )
    out = []
    for pos, value in zip(positions, values, strict=True):
        xy = (pos, value) if orient == "v" else (value, pos)
        out.append(
            ax.annotate(
                fmt.format(value),
                xy=xy,
                xytext=offset,
                textcoords="offset points",
                **align,
                **defaults,
            )
        )
    return out


def footnote(fig, text):
    """A caption line under the whole figure. Uses supxlabel so constrained
    layout reserves room for it -- a raw fig.text at a hand-tuned negative y
    is what detaches from the panels when the figure is resized.
    """
    return fig.supxlabel(text, fontsize=7, color=INK["muted"], style="italic")


if __name__ == "__main__":
    # Throwaway validation render (not committed): sanity-check panel-label
    # placement and font rendering before wiring this into real figures.
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        ax.plot([0, 1], [0, 1], color=CAT_COLORS[i % len(CAT_COLORS)])
        panel_label(ax, chr(ord("a") + i))
    plt.tight_layout()
    plt.savefig("_style_validation.png", bbox_inches="tight")
    print("Wrote _style_validation.png -- inspect, then delete.")
