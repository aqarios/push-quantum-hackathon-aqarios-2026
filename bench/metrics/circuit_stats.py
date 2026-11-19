"""Metric: circuit statistics that ``bench.execution.RunLog`` stored in ``solution.runtime``."""

import math

from luna_bench.custom import BaseMetric, FeatureResultContainer, MetricResult, metric, plot
from luna_bench.plots import Figure, MetricDimension
from luna_bench.plots.generics.metric_bar_plot import MetricBarPlot
from luna_model import Solution, Timing


class CircuitStatsResult(MetricResult):
    """Largest executed circuit and effort of one run; ``None`` for runs without a RunLog (e.g. FlexQAOA).

    ``depth``, ``two_qubit_gates``, ``total_gates`` and ``physical_qubits`` refer to the circuit transpiled
    onto the metrics backend (FakeHanoiV2, SWAPs included), the ``ideal_*`` values to the simulated circuit
    on the ideal backend (all-to-all coupling). ``qubits`` are the logical qubits (``MAX_QUBITS`` rule).
    """

    qubits: int | None = None
    physical_qubits: int | None = None
    depth: int | None = None
    two_qubit_gates: int | None = None
    total_gates: int | None = None
    ideal_depth: int | None = None
    ideal_two_qubit_gates: int | None = None
    ideal_total_gates: int | None = None
    circuits: int | None = None
    shots_per_circuit: int | None = None
    total_shots: int | None = None
    optimizer_iterations: int | None = None
    metrics_seconds: float | None = None


def largest(timing: Timing, key: str) -> int | None:
    """Maximum of the per-circuit values of ``key``; ``None`` if there are none (or none is a number)."""
    values = [value for value in timing.get(key) if not math.isnan(value)]
    return int(max(values)) if values else None


@metric
class CircuitStats(BaseMetric[CircuitStatsResult]):
    """Circuit statistics of a run of one of the local algorithms.

    Size values (qubits, depth, gates) are the maxima over all circuits a run executed. ``physical_qubits``
    counts the device qubits the transpiled circuit acts on, which can be more than the logical qubits
    when routing uses extra qubits. ``metrics_seconds`` is the time spent on the metrics transpilation,
    which is not part of the runtime.
    """

    def run(self, solution: Solution, feature_results: FeatureResultContainer) -> CircuitStatsResult:  # noqa: ARG002
        """Read the RunLog entries of ``solution.runtime``."""
        timing = solution.runtime
        if timing is None or not timing.get("depth"):
            return CircuitStatsResult()
        return CircuitStatsResult(
            qubits=largest(timing, "num_qubits"),
            physical_qubits=largest(timing, "physical_qubits"),
            depth=largest(timing, "depth"),
            two_qubit_gates=largest(timing, "two_qubit_gates"),
            total_gates=largest(timing, "total_gates"),
            ideal_depth=largest(timing, "ideal_depth"),
            ideal_two_qubit_gates=largest(timing, "ideal_two_qubit_gates"),
            ideal_total_gates=largest(timing, "ideal_total_gates"),
            circuits=len(timing.get("depth")),
            shots_per_circuit=int(timing.get("shots")[-1]),
            total_shots=int(sum(timing.get("shots"))),
            optimizer_iterations=int(timing.total_for("optimizer_iterations") or 0),
            metrics_seconds=float(sum(timing.get("metrics_seconds"))),
        )


@plot(CircuitStats)
class CircuitStatsPlot(MetricBarPlot):
    """Bar chart of one :class:`CircuitStats` field, chosen with ``y``, e.g.
    ``CircuitStatsPlot(y=MetricDimension("depth", "Depth"))``.

    Runs without circuit statistics (FlexQAOA, failed runs) are marked as missing.
    """

    figure: Figure = Figure(filename="two_qubit_gates", title="Two-qubit gates (largest circuit)")
    y: MetricDimension = MetricDimension("two_qubit_gates", "Two-qubit gates", baseline=0.0)
