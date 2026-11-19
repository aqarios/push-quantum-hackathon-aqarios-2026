"""Smoke tests: every team's algorithms run, discovery rejects a copied algorithm id and keeps at most
MAX_ALGORITHMS_PER_TEAM per team, the size classes cover every instance, the layer TTS counts the final circuit, a run
that breaks the time or qubit limit or the final-shot rule, or raises an error, ends without hanging the benchmark.

    uv run pytest
"""

import pickle
import time

import pytest
from luna_bench.custom import FeatureResultContainer
from luna_bench.features import OptSolFeature
from luna_model import Model, Sense, Solution, Vtype
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from bench import config, rules
from bench.building_blocks import lp_to_qubo
from bench.execution import ExecutionService, HackathonParticipantAlgorithm, RunLog, limits, layer_count
from bench.metrics import LayerTts
from bench.registry import TeamError, discover
from bench.utils import backtransform, load_instances, qubo_size, size_class

TEAM_ALGORITHMS = discover()


def maxcut_triangle() -> Model:
    """Max-cut on a triangle: the best cut has 2 edges."""
    model = Model(sense=Sense.MAX)
    x = [model.add_variable(f"x{i}", vtype=Vtype.BINARY) for i in range(3)]
    model.objective = sum(x[i] + x[j] - 2 * x[i] * x[j] for i, j in ((0, 1), (1, 2), (0, 2)))
    return model


@pytest.mark.parametrize("algorithm", TEAM_ALGORITHMS.values(), ids=TEAM_ALGORITHMS.keys())
def test_every_algorithm_returns_samples_and_circuit_statistics(algorithm):
    solution = algorithm.run(maxcut_triangle())
    assert len(solution.samples) > 0
    assert max(solution.obj_values) == pytest.approx(2.0)  # the optimum is among the samples
    assert solution.runtime.get("qpu_seconds")  # estimated QPU time for the metrics


def test_discovery_rejects_a_team_that_kept_the_example_algorithm_id(tmp_path, monkeypatch):
    """luna_bench registers only the first class of an algorithm id, so a copied template that kept "example.*" would
    run, and be reported, as the example team's class."""
    team = tmp_path / "teams_under_test" / "copycat"
    team.mkdir(parents=True)
    (team.parent / "__init__.py").write_text("")
    (team / "__init__.py").write_text(
        "from bench.execution import algorithm\n"
        "from hackathon.teams.example.my_algorithm import MyAlgorithm\n"
        "@algorithm(algorithm_id='example.CopiedTemplate')\n"
        "class CopiedTemplate(MyAlgorithm):\n"
        "    pass\n"
        "ALGORITHMS = {'Copied': CopiedTemplate()}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(TeamError, match="copycat.*example.CopiedTemplate"):
        discover("teams_under_test")


def test_layer_tts_counts_the_final_circuit():
    """Two runs of a gate with the same name and qubits but a different definition: the last one counts."""
    model = maxcut_triangle()
    _, transformation = lp_to_qubo(model)
    log = RunLog()
    execution_service = ExecutionService(log)
    pseudo_optimizer_iterations = 0
    for terms in (["ZII"], ["ZZI", "IZZ", "ZIZ"]):  # H layer + 1 layer, then H layer + 3 overlapping RZZ
        circuit = QuantumCircuit(3)
        circuit.h(range(3))
        circuit.append(PauliEvolutionGate(SparsePauliOp(terms), time=0.3), range(3))
        circuit.measure_all()
        counts = execution_service.execute(circuit, shots=256)
        pseudo_optimizer_iterations += 1
    solution = execution_service.execute_last_and_return_solution(transformation, model, circuit.copy(), pseudo_optimizer_iterations, None)
    optimum = config.FEATURES["optimum"]
    features = FeatureResultContainer.model_construct(data={OptSolFeature: {"optimum": (optimum.run(model), optimum)}})
    assert LayerTts().run(solution, features).layers == 4


def test_run_over_the_time_limit_returns_no_samples_but_the_circuit_statistics(monkeypatch):
    """The limit covers the whole run: here it is broken after the last circuit, in the post-processing."""

    class SlowPostProcessing(HackathonParticipantAlgorithm):
        def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
            _, transformation = lp_to_qubo(model)
            circuit = QuantumCircuit(3, 3)
            circuit.h(range(3))
            circuit.measure(range(3), range(3))
            counts = execution_service.execute(circuit, shots=64)
            time.sleep(0.5)
            return backtransform(counts, transformation, model)

    monkeypatch.setenv("BENCH_TIME_LIMIT", "0.2")
    solution = SlowPostProcessing().run(maxcut_triangle())
    assert len(solution.samples) == 0
    assert len(solution.runtime.get("depth")) == 1  # the executed circuit is still reported
    assert solution.runtime.total > 0.2


def test_circuit_over_the_qubit_limit_ends_the_run_with_an_empty_solution():
    class TooWide(HackathonParticipantAlgorithm):
        def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
            qubits = limits()[0] + 1
            circuit = QuantumCircuit(qubits)
            circuit.h(range(qubits))
            circuit.measure_all()
            execution_service.execute(circuit, shots=1)
            pytest.fail("the qubit limit was not enforced")

    assert len(TooWide().run(maxcut_triangle()).samples) == 0


def test_luna_model_error_fails_the_run_with_an_error_luna_bench_can_report():
    """luna_bench pickles the error of a failed run in its worker; luna_model's own errors cannot be pickled."""

    class UnknownVariable(HackathonParticipantAlgorithm):
        def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
            model.get_variable("not_in_the_model")
            pytest.fail("get_variable did not raise")

    with pytest.raises(RuntimeError, match="not_in_the_model") as error:
        UnknownVariable().run(maxcut_triangle())
    pickle.loads(pickle.dumps(error.value))


def test_every_instance_has_a_size_class_within_the_qubit_limit():
    sizes = {problem["name"]: qubo_size(model) for problem, model in load_instances()}
    assert max(sizes.values()) <= min(rules.MAX_QUBITS, max(rules.SIZES.values()))
    classes = {size_class(size) for size in sizes.values()}
    assert classes == set(rules.SIZES)  # instances in every size class


def _three_layers(n=4):
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return circuit


def test_layer_count_sees_through_nested_gates():
    nested = _three_layers()
    for _ in range(25):   # the old counter gave up after 20 rounds and reported 1 layer
        outer = QuantumCircuit(nested.num_qubits)
        outer.append(nested.to_gate(), range(nested.num_qubits))
        nested = outer
    assert layer_count(nested) == layer_count(_three_layers()) == 3


@pytest.mark.parametrize("name", ["measure", "barrier", "delay"])
def test_layer_count_skips_by_type_not_by_name(name):
    gate = Gate(name, 4, [])
    gate.definition = _three_layers()
    circuit = QuantumCircuit(4)
    circuit.append(gate, range(4))
    circuit.measure_all()   # real measurements are still not counted
    assert layer_count(circuit) == 3


def test_layer_count_opens_a_custom_gate_with_a_standard_name():
    body = QuantumCircuit(2)
    for _ in range(10):
        body.h(0)
        body.cx(0, 1)
    fake = Gate("rzz", 2, [0.1])
    fake.definition = body
    circuit = QuantumCircuit(2)
    circuit.append(fake, [0, 1])
    assert layer_count(circuit) == 20


def test_layer_count_rejects_control_flow():
    circuit = QuantumCircuit(2, 2)
    with circuit.for_loop(range(5)):
        circuit.h(0)
    with pytest.raises(ValueError, match="control flow"):
        layer_count(circuit)

# --------------------------------------------------------------------------------------
# Final shot: exactly rules.FINAL_SHOTS shots, executed once and last, are the solution
# --------------------------------------------------------------------------------------


def uniform_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    circuit.measure_all()
    return circuit


class FinalShotProbe(HackathonParticipantAlgorithm):
    """Executes ``before`` optimization circuits, then the final circuit ``finals`` times (through
    execute_last_and_return_solution, or through execute_final if ``via_execute_final``), then ``after`` more
    circuits."""

    before: int = 2
    finals: int = 1
    via_execute_final: bool = False
    after: int = 0
    drop_samples: bool = False

    def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
        _, transformation = lp_to_qubo(model)
        for _ in range(self.before):
            execution_service.execute(uniform_circuit(), shots=100)  # optimization: any number of shots
        solution = None
        for _ in range(self.finals):
            if self.via_execute_final:
                counts = execution_service.execute_final(uniform_circuit())
                if self.drop_samples:  # keep only the most frequent bitstring
                    counts = dict([max(counts.items(), key=lambda item: item[1])])
                solution = backtransform(counts, transformation, model)
            else:
                solution = execution_service.execute_last_and_return_solution(
                    transformation, model, uniform_circuit(), 0)
        for _ in range(self.after):
            execution_service.execute(uniform_circuit(), shots=100)
        if solution is None:  # no final circuit: a solution from the optimization's counts
            solution = backtransform(execution_service.execute(uniform_circuit(), shots=rules.FINAL_SHOTS),
                                     transformation, model)
        return solution


@pytest.mark.parametrize("via_execute_final", [False, True], ids=["execute_last", "execute_final"])
def test_final_circuit_with_final_shots_is_the_solution(via_execute_final):
    solution = FinalShotProbe(via_execute_final=via_execute_final).run(maxcut_triangle())
    assert int(solution.counts.sum()) == rules.FINAL_SHOTS
    assert list(solution.runtime.get("shots")) == [100, 100, rules.FINAL_SHOTS]


@pytest.mark.parametrize("probe", [
    FinalShotProbe(finals=0),  # no final circuit
    FinalShotProbe(finals=2),  # best of two final circuits
    FinalShotProbe(after=1),  # a circuit after the final one
    FinalShotProbe(via_execute_final=True, drop_samples=True),  # not all final shots in the solution
], ids=["no-final", "two-finals", "circuit-after", "samples-dropped"])
def test_final_shot_violation_ends_the_run_with_an_empty_solution(probe):
    solution = probe.run(maxcut_triangle())
    assert len(solution.samples) == 0
    assert solution.runtime.get("shots")  # the executed circuits are still reported


def test_discovery_keeps_the_first_algorithms_by_name_except_for_the_example_team(tmp_path, monkeypatch):
    root = tmp_path / "teams_capped"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for team in ("busy", "example"):
        (root / team).mkdir()
        (root / team / "__init__.py").write_text(
            "from bench.execution import algorithm\n"
            "from hackathon.teams.example.my_algorithm import MyAlgorithm\n"
            f"@algorithm(algorithm_id='{team}.Copy')\n"
            "class Copy(MyAlgorithm):\n"
            "    pass\n"
            "ALGORITHMS = {name: Copy() for name in ('zeta', 'alpha', 'gamma', 'beta')}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    names = set(discover("teams_capped"))
    assert {name for name in names if name.startswith("busy-")} == {"busy-alpha", "busy-beta"}
    assert len([name for name in names if name.startswith("example-")]) == 4
