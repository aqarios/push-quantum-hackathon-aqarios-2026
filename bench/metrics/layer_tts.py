"""Metric: time to solution in circuit layers, as FlexQAOA reports it (metadata "metrics"/"tts").

    TTS = L * ceil(log(1 - 0.99) / log(1 - p))

with ``L`` the layers of the final circuit on an all-to-all device (``bench.execution.layer_count``; FlexQAOA:
``layers_total`` of its job) and ``p`` the share of optimal samples (against the optimum of OptSolFeature): the
depth of the circuit times the shots needed to sample the optimum once with 99 %. The optimization loop before
the final circuit is not counted. ``inf`` if the optimum was never sampled.
"""

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


class LayerTtsResult(MetricResult):
    """Time to solution in layers of a run; ``None`` for runs without layer data.

    Attributes
    ----------
    tts : float | None
        Layers of the final circuit times the shots needed for 99 %; ``inf`` if the optimum was never sampled.
    layers : float | None
        Layers of the final circuit.
    required_shots : float | None
        ``ceil(log(1 - target) / log(1 - p))``.
    probability_optimal : float | None
        Share of the returned samples that are optimal.
    """

    tts: float | None = None
    layers: float | None = None
    required_shots: float | None = None
    probability_optimal: float | None = None


@metric(OptSolFeature)
class LayerTts(BaseMetric[LayerTtsResult]):
    """Time to solution in circuit layers (see the module docstring)."""

    direction: ClassVar[MetricDirection] = MetricDirection.LOWER_IS_BETTER
    target_probability: float = 0.99
    abs_tol: float = 1e-6

    def run(self, solution: Solution, feature_results: FeatureResultContainer) -> LayerTtsResult:
        """Layers of the last circuit and the share of optimal samples."""
        timing = solution.runtime
        layers = [value for value in (timing.get("layers") or []) if value == value] if timing is not None else []  # noqa: PLR0124
        if not layers:
            return LayerTtsResult()
        samples = int(solution.counts.sum()) if len(solution.samples) else 0
        optimum = feature_results.first(feature_cls=OptSolFeature).global_best_sol
        feasible = solution.filter_feasible() if samples else None
        optimal = (int(feasible.counts[np.isclose(feasible.obj_values, optimum, atol=self.abs_tol)].sum())
                   if feasible is not None and len(feasible.samples) else 0)
        p = optimal / samples if samples else 0.0
        if p == 0:
            return LayerTtsResult(tts=math.inf, layers=layers[-1], required_shots=math.inf, probability_optimal=0.0)
        shots = 1 if p >= 1 else math.ceil(math.log(1 - self.target_probability) / math.log(1 - p))
        return LayerTtsResult(tts=layers[-1] * shots, layers=layers[-1], required_shots=float(shots),
                              probability_optimal=p)


@plot(LayerTts)
class LayerTtsPlot(MetricBarPlot):
    """Bar chart of :class:`LayerTts`; runs that never sampled the optimum are marked (default) or, with
    ``Missing(policy="max")``, the worst case. Runs without layer data are left out with ``skip_runs_without_data``."""

    figure: Figure = Figure(filename="tts", title="Time to solution in circuit layers")
    y: MetricDimension = MetricDimension("tts", "Layers x shots to solution (99 %)", baseline=0.0)
    skip_runs_without_data: bool = False

    def rows(self, benchmark_results: BenchmarkResultContainer) -> list[dict[str, Any]]:
        """One row per run; ``NaN`` (missing) for runs without layer data."""
        return [
            {"algorithm": algorithm, "model": model,
             self.y.column: float("nan") if result.layers is None else self.value(result)}
            for model, algorithm, result in benchmark_results.get_all_metrics_of_type(self.metric_cls)
            if result.layers is not None or not self.skip_runs_without_data
        ]
