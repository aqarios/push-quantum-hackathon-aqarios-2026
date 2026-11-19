"""Metric: shots to solution, the time to solution counted in shots, for every algorithm including FlexQAOA."""

import math
from typing import Any, ClassVar

import numpy as np
from luna_bench.custom import (
    BaseMetric,
    BenchmarkResultContainer,
    FeatureResultContainer,
    MetricDirection,
    MetricResult,
    metric,
    plot,
)
from luna_bench.features import OptSolFeature
from luna_bench.plots import Figure, MetricDimension
from luna_bench.plots.generics.metric_bar_plot import MetricBarPlot
from luna_model import Solution


class ShotsToSolutionResult(MetricResult):
    """Shots to solution of a run; ``None`` for runs without shot data.

    Attributes
    ----------
    shots_to_solution : float | None
        Shots needed to sample the optimum with ``target_probability``; ``inf`` if it was never sampled.
    total_shots : int | None
        Shots of all circuits of the run (optimization included).
    probability_optimal : float | None
        Share of the returned samples that are optimal.
    """

    shots_to_solution: float | None = None
    total_shots: int | None = None
    probability_optimal: float | None = None


@metric(OptSolFeature)
class ShotsToSolution(BaseMetric[ShotsToSolutionResult]):
    """Time to solution in shots: luna_bench's TTS formula with the shots of the run instead of its runtime::

        STS = total_shots / M * ceil(log(1 - target_probability) / log(1 - p))

    with ``M`` returned samples, of which a share ``p`` is optimal. Every circuit evaluation counts with its
    shots, as on hardware: for the local algorithms from ``solution.runtime.get("shots")`` (RunLog), for
    FlexQAOA ``(optimizer evaluations + 1) x shots`` from the solve job (``bench.utils.record_luna_shots``).
    On IBM hardware a shot takes the circuit duration plus a 250 us repetition delay, so STS is roughly the
    estimated QPU time divided by 250-450 us.
    """

    direction: ClassVar[MetricDirection] = MetricDirection.LOWER_IS_BETTER
    target_probability: float = 0.99
    abs_tol: float = 1e-6

    def run(self, solution: Solution, feature_results: FeatureResultContainer) -> ShotsToSolutionResult:
        """Count the shots of the run and the optimal samples."""
        timing = solution.runtime
        if timing is None or not timing.get("shots"):
            return ShotsToSolutionResult()
        total = int(sum(timing.get("shots")))
        samples = int(solution.counts.sum()) if len(solution.samples) else 0
        if samples == 0:  # no samples, e.g. a run stopped by a benchmark limit
            return ShotsToSolutionResult(shots_to_solution=math.inf, total_shots=total, probability_optimal=0.0)
        optimum = feature_results.first(feature_cls=OptSolFeature).global_best_sol
        feasible = solution.filter_feasible()
        optimal = (int(feasible.counts[np.isclose(feasible.obj_values, optimum, atol=self.abs_tol)].sum())
                   if len(feasible.samples) else 0)
        p = optimal / samples
        if p == 0:
            return ShotsToSolutionResult(shots_to_solution=math.inf, total_shots=total, probability_optimal=0.0)
        repetitions = 1 if p == 1 else math.ceil(math.log(1 - self.target_probability) / math.log(1 - p))
        return ShotsToSolutionResult(shots_to_solution=total / samples * repetitions, total_shots=total,
                                     probability_optimal=p)


@plot(ShotsToSolution)
class ShotsToSolutionPlot(MetricBarPlot):
    """Bar chart of :class:`ShotsToSolution`; runs that never sampled the optimum are marked (default) or, with
    ``Missing(policy="max")``, the worst case. Runs without shot data are left out with ``skip_runs_without_data``."""

    figure: Figure = Figure(filename="shots_to_solution", title="Shots to solution")
    y: MetricDimension = MetricDimension("shots_to_solution", "Shots to solution (99 %)", baseline=0.0)
    skip_runs_without_data: bool = False

    def rows(self, benchmark_results: BenchmarkResultContainer) -> list[dict[str, Any]]:
        """One row per run; ``NaN`` (missing) for runs without shot data."""
        return [
            {"algorithm": algorithm, "model": model,
             self.y.column: float("nan") if result.total_shots is None else self.value(result)}
            for model, algorithm, result in benchmark_results.get_all_metrics_of_type(self.metric_cls)
            if result.total_shots is not None or not self.skip_runs_without_data
        ]
