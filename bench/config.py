"""Benchmark configuration: seeds, features, metrics and figures. Fixed for all participants.

The qubit and time limit and the size classes are in ``bench/rules.py``.
"""

import os

from luna_bench.features import OptSolFeature
from luna_bench.metrics import FeasibilityRatio, Runtime, TimeToSolution
from luna_bench.plots import PERCENT, FeatureDimension, Figure, MetricDimension, Missing

from bench.features import QuboSize, UseCase
from bench.metrics import (
    ApproximationRatio,
    ApproximationRatioPlot,
    CircuitStats,
    CircuitStatsPlot,
    FeasibilityRatioPlot,
    LayerTts,
    LayerTtsPlot,
    QpuTime,
    QpuTimePlot,
    RuntimePlot,
    ShotsToSolution,
    ShotsToSolutionPlot,
    TimeToSolutionPlot,
)

# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------

# Seeds of the transpiler, the simulator and the metrics transpilation. None, the default, draws new random numbers
# in every run. run_benchmark.py --seed N sets all three to N through the environment variable BENCH_SEED, which
# luna_bench's worker reads too; set it that way instead of changing this file.
SEED: int | None = int(os.environ["BENCH_SEED"]) if os.environ.get("BENCH_SEED") else None
TRANSPILER_SEED: int | None = SEED
SIMULATOR_SEED: int | None = SEED
METRICS_SEED: int | None = SEED

# Circuits are simulated on the ideal backend (all-to-all coupling, no SWAPs). For the circuit statistics
# and the estimated QPU time, every executed circuit is additionally transpiled onto this real device
# (coupling map, native gates, SWAPs) with a fixed optimization level and seed. That circuit is not run.
METRICS_BACKEND = "FakeHanoiV2"
METRICS_OPTIMIZATION_LEVEL = 2

FEATURES = {
    "optimum": OptSolFeature(max_runtime=60),  # SCIP
    "use-case": UseCase(),
    "size": QuboSize(),
}

METRICS = {
    "approximation-ratio": ApproximationRatio(),
    "feasibility-ratio": FeasibilityRatio(),
    "time-to-solution": TimeToSolution(),
    "runtime": Runtime(),
    "qpu-time": QpuTime(),
    "shots-to-solution": ShotsToSolution(),
    "tts": LayerTts(),  # FlexQAOA's time to solution: layers (all-to-all) x shots needed for 99 %
    "circuit": CircuitStats(),
}

# --------------------------------------------------------------------------------------
# Figures, saved to reports/<name>/<figure>.png by bench.figures.draw_figures. One figure per metric
# (group): every column is one variant, drawn per use case, per problem size (QUBO variables) and as one
# bar per algorithm over all instances (the rows). "Solved only" variants leave unsolved runs out of the
# mean and mark how many there are (luna_bench default); "worst case" variants count them as the worst value.
# --------------------------------------------------------------------------------------

# Groups are drawn in the order of their first instance in bench/data/instances.yaml; the MaxCut instances
# come first and run through the size classes in increasing order.
BY_USE_CASE = FeatureDimension(UseCase, label="Use case")
BY_SIZE = FeatureDimension(QuboSize, label="QUBO variables")
RATIO = {"scale": PERCENT, "reference": PERCENT, "reference_label": "Optimal (100%)", "baseline": 0.0}
SECONDS = {"baseline": 0.0}
LOG_SECONDS = {"symlog_columns": [0, 1], "linthresh": 0.01}  # times span orders of magnitude


def panels(plot, title: str) -> list:  # noqa: ANN001 - any luna_bench plot
    """The plot grouped by use case, by problem size, and as one bar per algorithm over all instances."""
    return [
        plot.model_copy(update={"grouping": BY_USE_CASE, "figure": Figure(title=f"{title}, per use case", show=False)}),
        plot.model_copy(update={"grouping": BY_SIZE, "figure": Figure(title=f"{title}, per problem size", show=False)}),
        plot.model_copy(update={"grouping": None, "figure": Figure(title=f"{title}, all instances", show=False)}),
    ]


QPU_TTS = MetricDimension("time_to_solution", "Time to solution (s)", **SECONDS)

FIGURES = {
    "approximation_ratio": ("Approximation ratio", [
        panels(ApproximationRatioPlot(), "Feasible samples of solved runs (marked: unsolved)"),
        panels(
            ApproximationRatioPlot(missing=Missing(policy=0.0),
                                   y=MetricDimension("approximation_ratio_all", "Approximation ratio [%]", **RATIO)),
            "Per shot, infeasible samples and unsolved runs = 0"),
    ]),
    "feasibility_ratio": ("Share of feasible samples", [
        panels(FeasibilityRatioPlot(), "Solved runs (marked: unsolved)"),
        panels(FeasibilityRatioPlot(missing=Missing(policy=0.0)), "Unsolved runs = 0"),
    ]),
    "time_to_solution": ("Time to solution (99 %) from the wall-clock runtime", [
        panels(TimeToSolutionPlot(), "Marked: never sampled the optimum or unsolved"),
        panels(TimeToSolutionPlot(missing=Missing(policy="max")),
                                 "Never sampled the optimum or unsolved = worst"),
    ], LOG_SECONDS),
    "qpu_tts": ("Time to solution (99 %) from the estimated QPU time on FakeHanoiV2", [
        panels(QpuTimePlot(y=QPU_TTS), "Marked: never sampled the optimum or no circuit data"),
        panels(QpuTimePlot(y=QPU_TTS, missing=Missing(policy="max"), skip_runs_without_data=True),
                                 "Never sampled the optimum = worst (no circuit data: left out)"),
    ], LOG_SECONDS),
    "tts": ("Time to solution in circuit layers (all-to-all): layers of the final circuit x shots needed for 99 %", [
        panels(LayerTtsPlot(), "Marked: never sampled the optimum or no layer data"),
        panels(LayerTtsPlot(missing=Missing(policy="max"), skip_runs_without_data=True),
               "Never sampled the optimum = worst (no layer data: left out)"),
    ], {"symlog_columns": [0, 1]}),
    "shots_to_solution": ("Shots to solution (99 %): time to solution counted in shots, FlexQAOA included", [
        panels(ShotsToSolutionPlot(), "Marked: never sampled the optimum or no shot data"),
        panels(ShotsToSolutionPlot(missing=Missing(policy="max"), skip_runs_without_data=True),
               "Never sampled the optimum = worst (no shot data: left out)"),
    ], {"symlog_columns": [0, 1]}),
    "runtime": ("Runtime per run", [
        panels(RuntimePlot(), "Wall clock"),
        panels(QpuTimePlot(), "Estimated QPU time: shots x (circuit duration on FakeHanoiV2 + repetition delay)"),
    ], LOG_SECONDS),
    "circuits": ("Largest circuit per run", [
        panels(CircuitStatsPlot(), "Two-qubit gates on FakeHanoiV2 (SWAPs included)"),
        panels(CircuitStatsPlot(y=MetricDimension("depth", "Depth", **SECONDS)), "Depth on FakeHanoiV2"),
        panels(CircuitStatsPlot(y=MetricDimension("qubits", "Logical qubits", **SECONDS)),
                                 "Logical qubits (simulated, MAX_QUBITS rule)"),
    ], {"symlog_columns": [0, 1]}),
}
