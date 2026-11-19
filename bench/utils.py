"""Benchmark helpers: build the model set, add components to a luna_bench benchmark, filter solutions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from luna_bench import Benchmark, ModelSet
from luna_bench._internal.wrappers import LunaAlgorithmWrapper
from luna_model import Model, Solution
from luna_model.errors import VariableNotExistingError
from luna_model.transformation import PassManager, pipelines
from luna_model.translator import QuboTranslator
from luna_quantum.solve.interfaces.algorithm_i import IAlgorithm

from bench.building_blocks import counts_to_bits
from bench.rules import SIZES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from luna_model import Timing
    from luna_model.transformation import TransformationOutput

INSTANCES_FILE = Path(__file__).resolve().parent / "data" / "instances.yaml"

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def qubo_size(model: Model) -> int:
    """Binary variables of the model's QUBO (luna_model's default transformation, slack included)."""
    qubo_model = PassManager([pipelines.ToUnconstrainedBinaryPipeline()]).run(model).model
    return len(QuboTranslator.from_lm(qubo_model).variable_names)


def luna_layers(algorithm: Any, model: Model) -> float:  # noqa: ANN401
    """Layers on an all-to-all device of the final circuit of Luna's QAOA and VQE on ``model``, for
    ``bench.metrics.LayerTts`` (their solve jobs report none, unlike FlexQAOA); ``nan`` for other algorithms.

    Both run on the model's QUBO (one qubit per QUBO variable). QAOA, analytically: a Hadamard layer, then per
    repetition the linear terms (RZ, one layer), the couplings (RZZ, one layer per colour of an edge colouring
    of the QUBO's interaction graph) and the mixer (RX, one layer). VQE: the same Qiskit ansatz built here and
    counted with ``bench.execution.layer_count``.
    """
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    qubo_model = PassManager([pipelines.ToUnconstrainedBinaryPipeline()]).run(model).model
    matrix = np.asarray(QuboTranslator.from_lm(qubo_model).matrix)
    size = len(matrix)
    name = getattr(algorithm, "algorithm_name", "")
    if name == "QAOA":
        import rustworkx  # noqa: PLC0415

        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from_no_data([(i, j) for i in range(size) for j in range(i + 1, size)
                                      if matrix[i, j] != 0 or matrix[j, i] != 0])
        colours = len(set(rustworkx.graph_misra_gries_edge_color(graph).values()))
        return float(1 + algorithm.reps * (2 + colours))
    if name == "VQE":
        from qiskit.circuit import library  # noqa: PLC0415

        from bench.execution import layer_count  # noqa: PLC0415

        return float(layer_count(getattr(library, algorithm.ansatz)(size, **algorithm.ansatz_config)))
    return math.nan


def load_instances(tag: str | None = None, file: Path = INSTANCES_FILE) -> list[tuple[dict[str, Any], Model]]:
    """The entries of ``file`` (default ``bench/data/instances.yaml``; with ``tag``, only those carrying it) and
    their models, built with luna_usecases and named after the entry."""
    problems = yaml.safe_load(file.read_text(encoding="utf-8"))["problems"]
    instances = []
    for problem in problems:
        if tag is not None and tag not in problem.get("tags", []):
            continue
        module, collection, method = problem["generator"].rsplit(".", 2)
        generator = getattr(getattr(importlib.import_module(f"luna_usecases.{module}"), collection), method)
        instance = generator(**problem["params"]).instances[0]
        model = instance.formulation.formulate(instance.data)
        model.name = problem["name"]
        instances.append((problem, model))
    return instances


def size_class(size: int) -> str:
    """Size class of an instance with ``size`` QUBO variables (``bench.rules.SIZES``)."""
    return next(name for name, largest in SIZES.items() if size <= largest)


def build_modelset(name: str, size: str = "very-large", tag: str | None = None,
                   file: Path = INSTANCES_FILE) -> ModelSet:
    """Build the instances of ``file`` (default ``bench/data/instances.yaml``) up to the size class ``size`` (with
    ``tag``: only entries carrying it) with luna_usecases and put them into a ModelSet."""
    model_set = ModelSet.create(name)
    for _, model in load_instances(tag, file):
        qubo_variables = qubo_size(model)
        if qubo_variables > SIZES[size]:
            continue
        print(f"{model.name:16s} {model.num_variables:3d} variables {model.num_constraints:3d} constraints"
              f" -> {qubo_variables:3d} QUBO variables ({size_class(qubo_variables)})")
        model_set.add(model)
    return model_set


# --------------------------------------------------------------------------------------
# Benchmark components
# luna_bench stores every component with its results under its name. The helpers below make the
# benchmark contain exactly the given components: stored ones that are not given are removed (for an
# algorithm, with its results), and add_plot & co. keep a stored component of the same name, so a
# changed one is removed before it is added again.
# --------------------------------------------------------------------------------------


def _remove_others(stored: list, names: Any, remove: Any) -> None:  # noqa: ANN401
    for entry in [entry for entry in stored if entry.name not in names]:
        remove(entry)


def add_algorithms(bench: Benchmark, algorithms: dict[str, Any], rerun: tuple[str, ...] = (),
                   only: list[str] | None = None) -> None:
    """Use exactly these algorithms. One whose parameters changed, or that is named in ``rerun``, replaces the
    stored one and runs again; an unchanged one keeps its results, so re-running the benchmark does not repeat
    finished runs. With ``only``, the other algorithms are left as stored (or not added yet)."""
    _remove_others(bench.algorithms, algorithms, bench.remove_algorithm)
    for name, algorithm in algorithms.items():
        if only is not None and name not in only:
            continue
        stored = next((entry for entry in bench.algorithms if entry.name == name), None)
        if stored is not None and name in rerun:
            bench.remove_algorithm(stored)
        elif stored is not None:
            new = LunaAlgorithmWrapper.wrap(algorithm) if isinstance(algorithm, IAlgorithm) else algorithm
            if stored.algorithm.model_dump(mode="json") == new.model_dump(mode="json"):
                continue
            bench.remove_algorithm(stored)
        bench.add_algorithm(name, algorithm)


def add_features(bench: Benchmark, features: dict[str, Any]) -> None:
    """Use exactly these features; they are recomputed on the next run."""
    _remove_others(bench.features, {}, bench.remove_feature)
    for name, feature in features.items():
        bench.add_feature(name, feature)


def add_metrics(bench: Benchmark, metrics: dict[str, Any]) -> None:
    """Use exactly these metrics. Metrics are cheap, so they are always recomputed; this also drops
    results of algorithms that were replaced."""
    _remove_others(bench.metrics, {}, bench.remove_metric)
    for name, metric in metrics.items():
        bench.add_metric(name, metric)


# --------------------------------------------------------------------------------------
# Solutions
# --------------------------------------------------------------------------------------

def backtransform(
    counts: Mapping[str, int],
    transformation: TransformationOutput,
    model: Model,
    timing: Timing | None = None,
) -> Solution:
    """Map measured QUBO bitstrings back to the original model and evaluate them there.

    Slack variables are dropped, so objective values and feasibility refer to the original,
    constrained model.
    """
    names = QuboTranslator.from_lm(transformation.model).variable_names
    bits, freqs = counts_to_bits(counts, len(names))
    samples = [dict(zip(names, map(int, row), strict=True)) for row in bits]
    qubo_solution = Solution.from_dicts(samples, model=transformation.model, counts=freqs.tolist())
    solution = model.evaluate(transformation.record.backward(qubo_solution))
    if timing is not None:
        solution.runtime = timing
    return solution

def keep_model_variables(solution: Solution, model: Model) -> Solution:
    """Remove variables the model does not know from a solution, e.g. helper variables of an algorithm."""

    def known(name: str) -> bool:
        try:
            model.get_variable(name)
        except VariableNotExistingError:
            return False
        return True

    unknown = [name for name in solution.variable_names if not known(name)]
    if unknown:
        solution.remove_vars(unknown)
    return solution


def evaluate_model_variables_only() -> None:
    """Make every ``Model.evaluate`` call apply :func:`keep_model_variables` first.

    ``Model.evaluate`` is where every solution is checked against the original model: luna_quantum
    calls it for Luna algorithms, :func:`backtransform` for the local ones. Some algorithms return
    helper variables, e.g. FlexQAOA ``dummy_packing_element_*`` for set packing (luna_quantum 1.3);
    without this filter the evaluation raises and the run fails.

    Installed once, when the ``bench`` package is imported; safe to call again.
    """
    if getattr(Model.evaluate, "keeps_model_variables", False):
        return
    evaluate = Model.evaluate

    def evaluate_model_variables(self: Model, solution: Solution, /, tol: float | None = None) -> Solution:
        return evaluate(self, keep_model_variables(solution, self), tol)

    evaluate_model_variables.keeps_model_variables = True
    Model.evaluate = evaluate_model_variables


def record_luna_shots() -> None:
    """Make luna_bench store the shots of Luna algorithms (FlexQAOA, QAOA, VQE) in ``solution.runtime``, for
    ``bench.metrics.ShotsToSolution``: ``(optimizer evaluations + 1) x shots`` from the solve job's metadata;
    likewise the layers for ``bench.metrics.LayerTts`` (FlexQAOA: metadata, QAOA and VQE: ``luna_layers``).

    Every evaluation counts with its shots, as it would on hardware (the platform's simulator may evaluate
    the cost exactly). luna_bench fetches the results in the main process, where this patch applies.
    """
    from luna_bench._internal.wrappers.luna_quantum.algorithms import LunaAlgorithm  # noqa: PLC0415
    from luna_model import Timing  # noqa: PLC0415
    from luna_quantum.solve import SolveJob  # noqa: PLC0415
    from returns.pipeline import is_successful  # noqa: PLC0415

    if getattr(LunaAlgorithm.fetch_result, "records_shots", False):
        return
    fetch = LunaAlgorithm.fetch_result

    def fetch_with_shots(self, model: Model, retrieval_data: Any) -> Any:  # noqa: ANN001, ANN401
        result = fetch(self, model, retrieval_data)
        if is_successful(result) and retrieval_data.luna_id:
            solution = result.unwrap()
            metadata = SolveJob.get_by_id(retrieval_data.luna_id).metadata or {}
            optimizer = metadata.get("optimizer") or {}
            evaluations = int(optimizer.get("nfev") or 0) + 1
            shots = float(getattr(self, "shots", 0) or solution.counts.sum())
            layers = float((metadata.get("circuit") or {}).get("layers_total") or luna_layers(self, model))
            solution.runtime = Timing.from_dict(
                {"shots": [shots] * evaluations, "layers": [layers] * evaluations,  # layers: bench.metrics.LayerTts
                 "optimizer_iterations": float(optimizer.get("nit") or 0)},
                total=solution.runtime.total if solution.runtime is not None else 0.0)
        return result

    fetch_with_shots.records_shots = True
    LunaAlgorithm.fetch_result = fetch_with_shots
