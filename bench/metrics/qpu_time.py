"""Metric: estimated QPU time of a run and the time to solution computed from it."""

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


class QpuTimeResult(MetricResult):
    """Estimated QPU time of a run; ``None`` for runs without circuit statistics (e.g. FlexQAOA).

    Attributes
    ----------
    qpu_seconds : float | None
        Estimated time on the real device for all circuits of the run (optimization included).
    time_to_solution : float | None
        Time to sample the optimum with ``target_probability``, from ``qpu_seconds``; ``inf`` if
        the run never sampled the optimum.
    probability_optimal : float | None
        Share of the returned samples (shots) that are optimal.
    """

    qpu_seconds: float | None = None
    time_to_solution: float | None = None
    probability_optimal: float | None = None


@metric(OptSolFeature)
class QpuTime(BaseMetric[QpuTimeResult]):
    """Estimated QPU time from the RunLog (``solution.runtime.get("qpu_seconds")``) and the TTS from it.

    Each executed circuit costs ``shots * (circuit duration + repetition delay)``, with the duration of
    the circuit transpiled onto ``METRICS_BACKEND`` (FakeHanoiV2), see ``bench.execution.metrics_stats``
    and ``bench.execution.estimate_qpu_seconds``. The time to solution uses luna_bench's formula with
    this QPU time instead of the wall clock::

        TTS = qpu_seconds / M * ceil(log(1 - target_probability) / log(1 - p))

    with ``M`` returned samples, of which a share ``p`` is optimal.
    """

    direction: ClassVar[MetricDirection] = MetricDirection.LOWER_IS_BETTER
    target_probability: float = 0.99
    abs_tol: float = 1e-6

    def run(self, solution: Solution, feature_results: FeatureResultContainer) -> QpuTimeResult:
        """Sum the estimated QPU time of all circuits and compute the time to solution from it."""
        timing = solution.runtime
        if timing is None or not timing.get("qpu_seconds"):
            return QpuTimeResult()
        qpu_seconds = float(sum(timing.get("qpu_seconds")))
        optimum = feature_results.first(feature_cls=OptSolFeature).global_best_sol

        samples = int(solution.counts.sum()) if len(solution.samples) else 0
        if samples == 0:  # no samples, e.g. a run stopped by a benchmark limit
            return QpuTimeResult(qpu_seconds=qpu_seconds, time_to_solution=math.inf, probability_optimal=0.0)
        feasible = solution.filter_feasible()
        optimal = (int(feasible.counts[np.isclose(feasible.obj_values, optimum, atol=self.abs_tol)].sum())
                   if len(feasible.samples) else 0)
        if optimal == 0:
            return QpuTimeResult(qpu_seconds=qpu_seconds, time_to_solution=math.inf, probability_optimal=0.0)

        p = optimal / samples
        repetitions = 1 if p == 1 else math.ceil(math.log(1 - self.target_probability) / math.log(1 - p))
        return QpuTimeResult(qpu_seconds=qpu_seconds, time_to_solution=qpu_seconds / samples * repetitions,
                             probability_optimal=p)


@plot(QpuTime)
class QpuTimePlot(MetricBarPlot):
    """Bar chart of one :class:`QpuTime` field, by default the estimated QPU time of a run.

    For the time to solution use ``y=MetricDimension("time_to_solution", ...)``: runs that never
    sampled the optimum are then marked (default) or, with ``Missing(policy="max")``, the worst case.
    Runs without circuit statistics (FlexQAOA) are marked as missing; with ``skip_runs_without_data``
    they are left out instead, e.g. for a worst-case plot, where missing data would count as worst.
    """

    figure: Figure = Figure(filename="qpu_time", title="Estimated QPU time per run")
    y: MetricDimension = MetricDimension("qpu_seconds", "Estimated QPU time (s)", baseline=0.0)
    skip_runs_without_data: bool = False

    def rows(self, benchmark_results: BenchmarkResultContainer) -> list[dict[str, Any]]:
        """One row per run; ``NaN`` (missing) for runs without an estimated QPU time."""
        return [
            {"algorithm": algorithm, "model": model,
             self.y.column: float("nan") if result.qpu_seconds is None else self.value(result)}
            for model, algorithm, result in benchmark_results.get_all_metrics_of_type(self.metric_cls)
            if result.qpu_seconds is not None or not self.skip_runs_without_data
        ]
