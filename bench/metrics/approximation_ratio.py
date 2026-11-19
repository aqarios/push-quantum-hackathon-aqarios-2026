"""Metric: approximation ratio in [0, 1] against the optimum, with and without infeasible samples."""

from typing import ClassVar

import numpy as np
from luna_bench.custom import BaseMetric, FeatureResultContainer, MetricDirection, MetricResult, metric, plot
from luna_bench.features import OptSolFeature
from luna_bench.plots import PERCENT, Figure, MetricDimension
from luna_bench.plots.generics.metric_bar_plot import MetricBarPlot
from luna_model import Solution

from .plots import ShowFailedRuns


class ApproximationRatioResult(MetricResult):
    """Approximation ratio of a run, averaged over its samples (shots).

    Attributes
    ----------
    approximation_ratio : float | None
        Mean over the feasible samples only; ``None`` if the run has no feasible sample.
    approximation_ratio_all : float
        Mean over all samples, infeasible samples count as 0.
    feasible_samples : int
        Number of feasible samples (shots) the ratio is based on.
    """

    approximation_ratio: float | None = None
    approximation_ratio_all: float = 0.0
    feasible_samples: int = 0


@metric(OptSolFeature)
class ApproximationRatio(BaseMetric[ApproximationRatioResult]):
    """Approximation ratio in ``[0, 1]`` against the optimum from ``OptSolFeature`` (SCIP).

    Each feasible sample ``x`` scores ``r(x) = max(0, 1 - |f(x) - f_opt| / |f_opt|)``, with ``f`` the
    objective of the original, constrained model, so ``1`` means optimal. For a maximization
    problem with a positive optimum, ``r(x) = f(x) / f_opt``. The result holds the mean over the
    feasible samples and the mean over all samples with infeasible ones counted as ``0``.
    ``f_opt`` must not be zero.
    """

    direction: ClassVar[MetricDirection] = MetricDirection.HIGHER_IS_BETTER

    def run(self, solution: Solution, feature_results: FeatureResultContainer) -> ApproximationRatioResult:
        """Average the per-sample ratios, weighted by how often each sample was measured."""
        optimum = feature_results.first(feature_cls=OptSolFeature).global_best_sol
        total = int(solution.counts.sum()) if len(solution.samples) else 0
        if total == 0:  # no samples, e.g. a run stopped by a benchmark limit
            return ApproximationRatioResult()
        feasible = solution.filter_feasible()
        num_feasible = int(feasible.counts.sum()) if len(feasible.samples) else 0
        if num_feasible == 0:
            return ApproximationRatioResult()
        ratios = np.clip(1 - np.abs(feasible.obj_values - optimum) / abs(optimum), 0.0, 1.0)
        weighted = float(ratios @ feasible.counts)
        return ApproximationRatioResult(approximation_ratio=weighted / num_feasible,
                                        approximation_ratio_all=weighted / total,
                                        feasible_samples=num_feasible)


@plot(ApproximationRatio)
class ApproximationRatioPlot(ShowFailedRuns, MetricBarPlot):
    """Bar chart of :class:`ApproximationRatio`.

    By default it shows the ratio over feasible samples; runs without any feasible sample and failed
    runs are left out of the mean and marked with their count. For the worst-case view use
    ``y=MetricDimension("approximation_ratio_all", ...)`` with ``missing=Missing(policy=0.0)``.
    """

    figure: Figure = Figure(filename="approximation_ratio", title="Approximation ratio, feasible samples only")
    y: MetricDimension = MetricDimension(
        "approximation_ratio", "Approximation ratio [%]", scale=PERCENT, reference=PERCENT,
        reference_label="Optimal (100%)", baseline=0.0)
