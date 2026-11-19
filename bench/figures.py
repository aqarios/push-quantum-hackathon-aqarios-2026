"""Figures with a grid of plots each, e.g. per metric: columns = variants, rows = per use case / size / overall."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from luna_bench.custom import BenchmarkResultContainer
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from luna_bench import Benchmark

PANEL_SIZE = (8.0, 4.0)  # inches per panel
DPI = 150
MAX_HORIZONTAL_LABELS = 6  # more algorithms per panel: tilted x labels
MAX_LABEL_CHARACTERS = 50  # longer x labels together (e.g. "RIW-LR-QAOA-XY"): tilted as well


def merged_results(benches: list[Benchmark], algorithms: set[str]) -> BenchmarkResultContainer:
    """The results of ``algorithms`` from several benchmarks (with different models) in one container."""
    parts = [BenchmarkResultContainer.from_benchmark(bench) for bench in benches]

    def keep(per_algorithm: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in per_algorithm.items() if name in algorithms}

    return BenchmarkResultContainer.model_construct(
        features={model: data for part in parts for model, data in part.features.items()},
        metrics={model: keep(data) for part in parts for model, data in part.metrics.items()},
        algorithms={model: keep(data) for part in parts for model, data in part.algorithms.items()},
    )


def draw_figures(bench: Benchmark | BenchmarkResultContainer, figures: dict[str, tuple],
                 directory: Path) -> list[Path]:
    """Draw every figure of ``figures`` (name -> (title, columns of plots[, options])) as ``<directory>/<name>.png``.

    ``bench`` is a benchmark or the results of several (:func:`merged_results`).

    Each plot is a luna_bench plot (e.g. ``ApproximationRatioPlot``) and is drawn into its panel with
    its own settings; its figure title becomes the panel title. The plots of a row share their grouping,
    so every row gets one legend right of it. Options: ``{"symlog_columns": [0, 1], "linthresh": 1.0}`` draws
    these columns with a symmetric-log y axis (linear below ``linthresh``, so zero and missing-value markers
    stay visible), for values spanning orders of magnitude.
    """
    results = bench if isinstance(bench, BenchmarkResultContainer) else BenchmarkResultContainer.from_benchmark(bench)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, (title, columns, *options) in figures.items():
        options = options[0] if options else {}
        symlog_columns, linthresh = options.get("symlog_columns", []), options.get("linthresh", 1.0)
        rows = list(zip(*columns, strict=True))
        figure, axes = plt.subplots(
            len(rows), len(columns), squeeze=False, dpi=DPI,
            figsize=(len(columns) * PANEL_SIZE[0], len(rows) * PANEL_SIZE[1]),
        )
        shown: set[str] = set()
        for row_axes, row in zip(axes, rows, strict=True):
            for column, (panel, plot) in enumerate(zip(row_axes, row, strict=True)):
                plot.draw_into(panel, results)
                if column in symlog_columns:
                    panel.set_yscale("symlog", linthresh=linthresh)
                    panel.set_ylim(0, 2 * panel.get_ylim()[1])  # headroom above the highest bar
                # Horizontal labels save height; with many or long names they would overlap, so they are tilted.
                labels = [label.get_text() for label in panel.get_xticklabels()]
                tilted = len(labels) > MAX_HORIZONTAL_LABELS or sum(map(len, labels)) > MAX_LABEL_CHARACTERS
                plt.setp(panel.get_xticklabels(), rotation=25 if tilted else 0,
                         horizontalalignment="right" if tilted else "center", rotation_mode="anchor")
                panel.set_xlabel("")
            _one_legend_per_row(row_axes, shown)
        figure.suptitle(title, fontsize="x-large")
        figure.tight_layout()
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(figure)
        files.append(path)
    return files


def _one_legend_per_row(row_axes: Any, shown: set[str]) -> None:  # noqa: ANN401 - matplotlib axes
    """Replace the legends of a row by one legend right of the row, so none covers the bars.

    It lists the groups of the row (e.g. the use cases) and the entries every plot draws (reference
    line, error bars, missing values) unless an earlier row of the figure (``shown``) has them already.
    """
    title, handles, labels = None, [], []
    for panel in row_axes:
        legend = panel.get_legend()
        if legend is None:
            continue
        title = title or legend.get_title().get_text()
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True):
            label = text.get_text()
            if label not in labels and not (_shared(label) and label in shown):
                handles.append(handle)
                labels.append(label)
        legend.remove()
    # Groups first, then the shared entries.
    order = sorted(range(len(labels)), key=lambda index: _shared(labels[index]))
    if order:
        row_axes[-1].legend([handles[i] for i in order], [labels[i] for i in order], title=title or None,
                            loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small", title_fontsize="small")
    shown.update(labels)


def _shared(label: str) -> bool:
    """Legend entries every plot draws, independent of the grouping."""
    return label.startswith(("Optimal", "Upper Limit", "±", "missing values"))


def tts_ranking(frame: Any, path: Path, column: str = "tts/tts") -> Path | None:  # noqa: ANN401 - pandas DataFrame
    """Bar chart of the time to solution in circuit layers (``column`` of results.csv) per algorithm: the geometric
    mean over the instances whose optimum was sampled, algorithms sorted by the number of unsolved instances (whose
    TTS is infinite), then by that mean. Algorithms without data are left out."""
    import numpy as np  # noqa: PLC0415
    from luna_bench.plots.utils import LunaColours  # noqa: PLC0415

    rows = []
    for algorithm, runs in frame.groupby("algorithm"):
        solved = runs["time-to-solution/probability_optimal"].fillna(0) > 0
        tts = runs.loc[solved, column] if column in runs else runs.loc[solved, "algorithm"] * float("nan")
        if solved.any() and tts.isna().all():  # no data
            continue
        tts = tts[(tts > 0) & np.isfinite(tts)]
        rows.append((int((~solved).sum()), 10 ** np.log10(tts).mean() if len(tts) else np.nan, algorithm,
                     int(solved.sum()), len(runs)))
    if not rows:
        return None
    most = max(row[4] for row in rows)  # algorithms that did not run on every instance come last, apart
    rows.sort(key=lambda row: (row[4] < most, row[0], np.inf if np.isnan(row[1]) else row[1]))
    figure, axis = plt.subplots(figsize=(max(8.0, 0.75 * len(rows)), 5.0), dpi=DPI)
    bars = axis.bar([row[2] for row in rows], [row[1] for row in rows], color=LunaColours.palette(num_colors=len(rows)))
    for bar, (_, mean, _, solved, total) in zip(bars, rows, strict=True):
        if total < most:
            bar.set_alpha(0.45)
            bar.set_hatch("//")
        if not np.isnan(mean):
            axis.annotate(f"{solved}/{total}", (bar.get_x() + bar.get_width() / 2, mean), ha="center", va="bottom",
                          fontsize="small", xytext=(0, 2), textcoords="offset points")
    if (complete := sum(row[4] == most for row in rows)) < len(rows):
        axis.axvline(complete - 0.5, color="grey", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylabel("Layers x shots to solution, geometric mean")
    axis.set_title("Time to solution in circuit layers (all-to-all) per algorithm, best first: fewest unsolved instances,\n"
                   "then the mean over the solved ones (label: solved / instances; hatched, right of the line: "
                   "not run on every instance)")
    plt.setp(axis.get_xticklabels(), rotation=25, horizontalalignment="right", rotation_mode="anchor")
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path
