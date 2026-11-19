"""Plots that also show unsolved runs, so that an unsolved problem always counts against the algorithm.

luna_bench computes no metric for a run that failed (e.g. a solve job that crashed), so by default
such a run silently disappears from every plot. :class:`ShowFailedRuns` adds it as a missing value
instead, and also turns runs without samples (stopped by a benchmark limit) into missing values. The
plot's ``missing`` policy then decides: marked and counted (the default), or filled with the worst
case (e.g. ``Missing(policy=0.0)`` for ratios, ``Missing(policy="max")`` for times).
"""

from typing import Any, ClassVar

from luna_bench.custom import BenchmarkResultContainer, plot
from luna_bench.metrics import FeasibilityRatio, Runtime, TimeToSolution
from luna_bench.plots import FeasibilityRatioPlot as LunaFeasibilityRatioPlot
from luna_bench.plots import RuntimePlot as LunaRuntimePlot
from luna_bench.plots import TimeToSolutionPlot as LunaTimeToSolutionPlot


class ShowFailedRuns:
    """Mixin for metric bar plots: failed runs, and runs without samples, become missing values.

    Set ``empty_runs_are_missing = False`` for plots of the effort (e.g. runtime), where a run stopped
    by a limit still has a valid value.
    """

    empty_runs_are_missing = True

    def rows(self, benchmark_results: BenchmarkResultContainer) -> list[dict[str, Any]]:
        """The plot's rows, with a missing value for every failed run and every run without samples."""
        rows = super().rows(benchmark_results)
        runs = list(benchmark_results.get_all_algorithms())
        if self.empty_runs_are_missing:
            empty = {(algorithm, model) for model, algorithm, run in runs
                     if run.solution is None or len(run.solution.samples) == 0}
            for row in rows:
                if (row["algorithm"], row["model"]) in empty:
                    row[self.y.column] = float("nan")
        plotted = {(row["algorithm"], row["model"]) for row in rows}
        rows += [{"algorithm": algorithm, "model": model, self.y.column: float("nan")}
                 for model, algorithm, _ in runs if (algorithm, model) not in plotted]
        return rows


@plot(FeasibilityRatio)
class FeasibilityRatioPlot(ShowFailedRuns, LunaFeasibilityRatioPlot):
    """luna_bench's feasibility ratio plot, unsolved runs included."""


@plot(TimeToSolution)
class TimeToSolutionPlot(ShowFailedRuns, LunaTimeToSolutionPlot):
    """luna_bench's time-to-solution plot, unsolved runs included."""


@plot(Runtime)
class RuntimePlot(ShowFailedRuns, LunaRuntimePlot):
    """luna_bench's runtime plot; failed runs are missing, runs stopped by a limit show their runtime."""

    empty_runs_are_missing: ClassVar[bool] = False
