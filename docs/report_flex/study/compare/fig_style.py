"""One consistent figure style for compare/fig_*.pdf|png (matplotlib). Categorical colours: fixed order of the
reference palette (dataviz skill, references/palette.md, light mode); identity is also carried by marker shape."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# fixed identity -> colour/marker (never by rank)
VARIANT = {"full_corr": (SERIES[0], "o", "full, F1-corrected"), "trilinear": (SERIES[1], "s", "trilinear"),
           "quadratic": (SERIES[2], "^", "quadratic"), "full_raw": (SERIES[3], "D", "full, uncorrected (robot now)")}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5, "legend.frameon": False, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    fig.savefig(path_noext + ".png", bbox_inches="tight")
    plt.close(fig)
