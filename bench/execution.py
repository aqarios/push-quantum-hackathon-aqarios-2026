"""Circuit execution for all algorithms: registration, ideal backend, noiseless simulation, run log, limits.

Every algorithm is registered with :func:`algorithm` and runs its circuits through :func:`ExecutionService.execute`,
which applies the benchmark rules of ``bench/rules.py`` to every algorithm alike: a run whose circuits
need more than ``MAX_QUBITS`` logical qubits, or that takes longer than ``TIME_LIMIT`` seconds, stops.
It then ends with an empty solution: no samples, so it counts as not solved, but with the statistics
of its circuits (wall clock, estimated QPU time, depth, gates, qubits).

Circuits are simulated without noise as statevectors: a circuit is simulated once and all shots are
sampled from its final state. On the ``"ideal"`` backend (every qubit coupled to every other one, the
native gates of IBM's Falcon devices) transpiling adds no SWAPs, so only the logical qubits are simulated.

For the statistics, every executed circuit is additionally transpiled onto ``METRICS_BACKEND``
(FakeHanoiV2: real coupling map, native gates, SWAPs), see :func:`metrics_stats`. Depth, gate counts and
the estimated QPU time refer to that circuit; the ideal-backend values are kept as ``ideal_*``. The
metrics transpilation is cached by circuit structure and does not count toward the run's time.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from luna_bench.custom import BaseAlgorithmSync
from luna_bench.custom import algorithm as luna_bench_algorithm
from luna_model import Solution, Timing
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ControlFlowOp, Delay, Measure, Parameter
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target, generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

from bench import config, rules
from bench.utils import backtransform

if TYPE_CHECKING:
    from collections.abc import Mapping

    from luna_model import Model

logger = logging.getLogger(__name__)

class HackathonParticipantAlgorithm(BaseAlgorithmSync, ABC):
    """Algorithm created from hackathon participants"""
    @abstractmethod
    def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
        """
        Method signature that participants have to follow

        Parameters
        ----------
        model
            Encoded constrained optimization problem
        execution_service
            ExecutionService instance to execute circuits in an unified and correct for benchmarking way
            also to track time and logs
        Returns
        -------
        Solution
            Result of optimization, that will be interpreted from Luna bench
        """

    def run(self, model: Model) -> Solution:
        """
        Wraps standard BaseAlgorithmSync run to apply the time limits.

        A run that breaks a benchmark limit returns an empty solution with the statistics of the
            circuits it executed instead of failing.
        Any other error fails the run as a ``RuntimeError`` that names the original error.
        Parameters
        ----------
        model
            Encoded constrained optimization problem
        Returns
        -------
        Solution
            Result of optimization, that will be interpreted from Luna bench
        """
        log = RunLog()
        execution_service = ExecutionService(log)
        try:
            solution = self.run_with_logging(model, execution_service)
            solution.runtime = log.timing()
            log.check_final(solution)
            return solution

        except LimitExceeded as error:
            logger.warning("%s stopped on %s: %s", type(self).__name__, model.name, error)
            solution = Solution.from_dicts([], model=model)
            solution.runtime = error.log.timing()
            return solution
        except Exception as error:
            # luna_model raises Rust errors that pickle cannot serialize. luna_bench pickles the error of a failed
            # run in its worker, so such an error would never be reported and the benchmark would hang.
            msg = f"{type(error).__name__}: {error}"
            raise RuntimeError(msg) from error

class LimitExceeded(RuntimeError):  # noqa: N818 - reads better than LimitExceededError
    """A run broke a benchmark limit; :meth:`HackathonParticipantAlgorithm.run` turns it into an empty solution."""

    def __init__(self, message: str, log: RunLog) -> None:
        super().__init__(message)
        self.log = log


class FinalShotViolation(LimitExceeded):  # noqa: N818
    """A run broke the final-shot rule (``rules.FINAL_SHOTS``); it ends with an empty solution like a broken limit."""


def algorithm(algorithm_id: str):  # noqa: ANN201 - returns a class decorator
    """Register an algorithm in luna_bench (like ``luna_bench.custom.algorithm``)"""

    def decorate(cls: type) -> type:
        if not issubclass(cls, HackathonParticipantAlgorithm) or cls.run is not HackathonParticipantAlgorithm.run:
            raise TypeError(f"{cls.__name__} is not a HackathonParticipantAlgorithm")
        return luna_bench_algorithm(algorithm_id=algorithm_id)(cls)

    return decorate

class IdealBackend(BackendV2):
    """Device with every qubit coupled to every other one and the native gates and median gate durations
    of an IBM fake backend. It is only a transpilation target: circuits are run by :func:`ExecutionService.execute`."""

    def __init__(self, num_qubits: int = rules.MAX_QUBITS, reference: str = "FakeHanoiV2") -> None:
        super().__init__(name="ideal")
        durations = getattr(fake_provider, reference)().target

        def properties(gate: str, qargs: list[tuple[int, ...]]) -> dict:
            duration = median(p.duration for p in durations[gate].values() if p is not None and p.duration is not None)
            return {q: InstructionProperties(duration=duration) for q in qargs}

        qubits = [(q,) for q in range(num_qubits)]
        pairs = [(a, b) for a in range(num_qubits) for b in range(num_qubits) if a != b]
        self._target = Target(num_qubits=num_qubits, dt=durations.dt)
        for gate in (IGate(), RZGate(Parameter("theta")), SXGate(), XGate()):
            self._target.add_instruction(gate, properties(gate.name, qubits))
        self._target.add_instruction(CXGate(), properties("cx", pairs))
        self._target.add_instruction(Measure(), properties("measure", qubits))

    @property
    def target(self) -> Target:  # noqa: D102
        return self._target

    @property
    def max_circuits(self) -> None:  # noqa: D102
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(self, run_input, **options):  # noqa: ANN001, ANN003, ANN201, ARG002, D102
        msg = "Run circuits with bench.execution.ExecutionService.execute"
        raise NotImplementedError(msg)


@functools.cache
def get_backend(name: str) -> BackendV2:
    """``"ideal"`` (:class:`IdealBackend`) or an IBM fake backend by class name, e.g. ``"FakeHanoiV2"``
    (see ``qiskit_ibm_runtime.fake_provider``). Circuits are simulated without noise on both."""
    if name == "ideal":
        return IdealBackend()
    return getattr(fake_provider, name)()


@functools.cache
def _simulator(backend: BackendV2) -> AerSimulator:  # noqa: ARG001 - the same noiseless simulator for every backend
    return AerSimulator(method="statevector",
                        seed_simulator=config.SIMULATOR_SEED)


def limits() -> tuple[int, float]:
    """``MAX_QUBITS`` and ``TIME_LIMIT`` of ``bench/rules.py``, unless overridden by the environment
    variables ``BENCH_MAX_QUBITS`` / ``BENCH_TIME_LIMIT`` (run_benchmark.py sets them for ``--time-limit``)."""
    return (int(os.environ.get("BENCH_MAX_QUBITS", rules.MAX_QUBITS)),
            float(os.environ.get("BENCH_TIME_LIMIT", rules.TIME_LIMIT)))

class ExecutionService():
    """ Execution Service used to transpile and run circuits in a consistent way across participants"""
    def __init__(self, log: RunLog) -> None:
        """
        Constructs an Instance of ExecutionService.
        1 Instance correspond to 1 algorithm and 1 Problem
        Parameters
        ----------
        """
        self._logical_circuit = None
        self._transpiled_parametrized_circuit = None
        self._backend = get_backend("ideal")
        self._log = log

    def execute(self, logical_circuit: QuantumCircuit, parameters: np.ndarray | Mapping[Parameter, float] | None = None,
                shots: int = 1024) -> dict[str, int]:
        """
        Executes a circuit, logs time and calculates statistics.
        All circuit executions for benchmark should be made through it.

        Parameters
        ----------
        logical_circuit
            Logical circuit to be transpiled, bound with parameters and executed
            If circuit is same, no additional transpilation will be performed
        parameters
            Values for the parameters of ``logical_circuit``: a dict ``{parameter: value}``, or an array in the
            order of ``logical_circuit.parameters``. That order is sorted by name, not by creation:
            ``Parameter("gamma")`` and ``Parameter("beta")`` bind as ``[beta, gamma]``. A ``ParameterVector``
            keeps its index order.
        shots
            Number of shots for simulation, could be changed until timeout is reached.
        Returns
        -------
        Counts for this specific execution
        """
        if self._logical_circuit is None or logical_circuit != self._logical_circuit:
            max_qubits, _ = limits()
            if logical_circuit.num_qubits > max_qubits:
                msg = f"Needs {logical_circuit.num_qubits} qubits, more than MAX_QUBITS={max_qubits}"
                raise LimitExceeded(msg, self._log)

            self._logical_circuit = logical_circuit.copy()
            self._transpiled_parametrized_circuit = generate_preset_pass_manager(optimization_level=2,backend=self._backend, seed_transpiler=config.TRANSPILER_SEED).run(self._logical_circuit)

        if parameters is not None:
            transpiled_bound_circuit = self._transpiled_parametrized_circuit.assign_parameters(parameters, inplace=False)
        else:
            transpiled_bound_circuit = self._transpiled_parametrized_circuit

        layers = layer_count(self._logical_circuit)

        start = time.perf_counter()
        counts = _simulator(self._backend).run(transpiled_bound_circuit, shots=shots).result().get_counts()
        self._log.add(transpiled_bound_circuit, shots, time.perf_counter() - start, self._backend, layers)
        self._log.check_time()
        return counts

    def execute_final(self, logical_circuit: QuantumCircuit,
                      parameters: np.ndarray | Mapping[Parameter, float] | None = None) -> dict[str, int]:
        """Execute the final circuit with exactly ``rules.FINAL_SHOTS`` shots and return its counts.

        Use it when you map the counts to a solution yourself (an encoding of your own); with a QUBO from
        ``lp_to_qubo``, :meth:`execute_last_and_return_solution` does both. A run calls exactly one of the two,
        once, and executes no circuit after it; its solution consists of exactly these shots.
        """
        counts = self.execute(logical_circuit, parameters=parameters, shots=rules.FINAL_SHOTS)
        self._log.final_calls += 1
        self._log.final_index = len(self._log.circuits) - 1
        return counts

    def execute_last_and_return_solution(self, transformation, model, logical_circuit: QuantumCircuit, optimizer_iterations : int,
                                         parameters: np.ndarray | Mapping[Parameter, float] | None = None) -> Solution:
        """Execute the final circuit (:meth:`execute_final`) and return its samples as a solution of ``model``.

        ``transformation`` is the second value of ``lp_to_qubo(model)``; it maps the QUBO bitstrings back to the
        model's variables. ``optimizer_iterations`` is reported in the results
        (:meth:`record_optimizer_iterations`); pass 0 without an optimizer.
        """
        self.record_optimizer_iterations(optimizer_iterations)
        counts = self.execute_final(logical_circuit, parameters=parameters)
        solution = backtransform(counts, transformation, model, self._log.timing())
        return solution

    def elapsed_time(self) -> float:
        """Wall-clock seconds since the log was created"""
        return self._log.elapsed()

    def remaining_time(self) -> float:
        """Seconds left until the time limit, to stop in time with the best result so far.

        Pass the method itself, not its value: ``minimize(cost, x0, time_left=execution_service.remaining_time)``.
        """
        return self._log.remaining()

    def record_optimizer_iterations(self, iterations: int) -> None:
        """Report the optimizer iterations of the run (``minimize`` returns them), not the cost evaluations:
        those are the executed circuits, which are counted anyway."""
        self._log.optimizer_iterations = iterations

# Qiskit's standard gates by name, layer_count counts only these as built.
_STANDARD_GATES = get_standard_gate_name_mapping()
# Deepest nesting of gate definitions layer_count follows before it rejects the circuit.
_MAX_NESTING = 200

def layer_count(circuit: QuantumCircuit) -> int:
    """Count the layers of ``circuit`` on an all-to-all device, as FlexQAOA's time to solution does.

    A layer is a set of gates on disjoint qubits. A Qiskit standard gate on at most two qubits counts as built
    (RZZ, RXX, CRY, ...: no decomposition into native gates, no routing). Every other operation is replaced by
    its definition until only such gates are left, however deeply it is nested and whatever it is named.
    Measurements, barriers and delays are not counted; they are recognized by type, not by name. Control flow
    and gates without a definition raise ``ValueError``.
    """
    flat = QuantumCircuit(circuit.num_qubits)

    def flatten(source: QuantumCircuit, qubits: list[int], level: int) -> None:
        if level > _MAX_NESTING:
            msg = f"Gate definitions are nested deeper than {_MAX_NESTING} levels"
            raise ValueError(msg)
        for inst in source.data:
            operation = inst.operation
            target = [qubits[source.find_bit(q).index] for q in inst.qubits]
            if isinstance(operation, (Barrier, Delay, Measure)):
                continue
            if isinstance(operation, ControlFlowOp):
                msg = f"Circuits with control flow ({operation.name}) are not supported"
                raise ValueError(msg)
            standard = _STANDARD_GATES.get(operation.name)
            if len(target) <= 2 and standard is not None and type(operation) is type(standard):
                flat.append(operation, target)
                continue
            if operation.definition is None:
                msg = f"Gate {operation.name!r} has no definition"
                raise ValueError(msg)
            flatten(operation.definition, target, level + 1)

    flatten(circuit, list(range(circuit.num_qubits)), 0)
    return flat.depth()


def circuit_stats(circuit: QuantumCircuit) -> dict[str, int]:
    """Depth, two-qubit gates, gates (measurements, barriers and delays not counted) and qubits acted on."""
    gates = [inst for inst in circuit.data if inst.operation.name not in ("barrier", "delay", "measure")]
    return {
        "depth": circuit.depth(lambda inst: inst.operation.name not in ("barrier", "delay")),
        "two_qubit_gates": sum(len(inst.qubits) == 2 for inst in gates),
        "total_gates": len(gates),
        "qubits": len({q for inst in gates for q in inst.qubits}),
    }


@functools.cache
def _metrics_pass_manager():  # noqa: ANN202 - a qiskit StagedPassManager
    return generate_preset_pass_manager(
        optimization_level=config.METRICS_OPTIMIZATION_LEVEL, backend=get_backend(config.METRICS_BACKEND),
        seed_transpiler=config.METRICS_SEED)


# Statistics of the metrics transpilation by circuit structure, see metrics_stats.
_metrics_cache: dict[tuple[int, int], dict[str, float]] = {}


def metrics_stats(circuit: QuantumCircuit) -> dict[str, float]:
    """Statistics of ``circuit`` transpiled onto ``METRICS_BACKEND``; the transpiled circuit is not run.

    Returns depth, two-qubit gates, gates, physical qubits and the duration in seconds (critical path with
    the device's gate and measurement durations). Circuits that differ only in their angles (e.g. QAOA in
    every optimizer iteration) share the result, which is computed once. Circuits with more active qubits
    than the device get ``NaN``, and so do circuits the transpiler fails on: the statistics must never
    stop a run.
    """
    structure = tuple((inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits))
                      for inst in circuit.data)
    key = (len(structure), hash(structure))
    if key not in _metrics_cache:
        active = active_qubits_only(circuit)
        device = get_backend(config.METRICS_BACKEND)
        stats = dict.fromkeys(("depth", "two_qubit_gates", "total_gates", "qubits", "duration"), math.nan)
        if active.num_qubits <= device.num_qubits:
            try:
                transpiled = _metrics_pass_manager().run(active)
                stats = {**circuit_stats(transpiled), "duration": transpiled.estimate_duration(device.target, unit="s")}
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:  # also Rust panics of the transpiler (pyo3 PanicException)
                logger.exception("Metrics transpilation onto %s failed", config.METRICS_BACKEND)
        _metrics_cache[key] = stats
    return _metrics_cache[key]


def active_qubits_only(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy of ``circuit`` on the qubits it acts on, renumbered from 0, with fresh registers.

    The ideal backend has ``MAX_QUBITS`` qubits, so a transpiled circuit carries idle ones that may not fit onto
    the metrics backend. (Removing them with ``DAGCircuit.remove_qubits`` instead makes Qiskit 2.5's layout
    stage panic on some circuits.)
    """
    active = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})
    position = {index: k for k, index in enumerate(active)}
    compact = QuantumCircuit(len(active), circuit.num_clbits, global_phase=circuit.global_phase)
    for inst in circuit.data:
        compact.append(inst.operation, [position[circuit.find_bit(q).index] for q in inst.qubits],
                       [circuit.find_bit(c).index for c in inst.clbits], copy=False)
    return compact


def estimate_qpu_seconds(duration: float, shots: int, backend: BackendV2) -> float:
    """Time a circuit of ``duration`` seconds would occupy ``backend``: ``shots * (duration + repetition delay)``.

    The repetition delay (250 us on IBM devices) is the wait between shots. Job submission and queueing
    are not included.
    """
    return shots * (duration + getattr(backend, "default_rep_delay", 250e-6))


class RunLog:
    """Statistics of one algorithm run: one entry per executed (transpiled) circuit.

    :meth:`timing` packs them into the :class:`luna_model.Timing` that is attached to the returned
    solution, so they are stored by luna_bench and available to its metrics:
    ``solution.runtime.total`` is the wall-clock time of the run without the metrics transpilation, and
    ``solution.runtime.get(key)`` gives one value per executed circuit for

    * ``depth``, ``two_qubit_gates``, ``total_gates``, ``physical_qubits``: on ``METRICS_BACKEND``
      (FakeHanoiV2, SWAPs included), see :func:`metrics_stats`;
    * ``qpu_seconds``: estimated time on that device, see :func:`estimate_qpu_seconds`;
    * ``ideal_depth``, ``ideal_two_qubit_gates``, ``ideal_total_gates``: on the simulated backend;
    * ``num_qubits``: logical qubits (the ones simulated and limited by ``MAX_QUBITS``);
    * ``shots``, ``execute`` (simulation seconds), ``metrics_seconds`` (metrics transpilation seconds).

    ``optimizer_iterations`` is one value per run, reported by the algorithm through
    :meth:`ExecutionService.record_optimizer_iterations` (0 if it does not).
    """

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.metrics_seconds = 0.0
        self.circuits: list[dict[str, float]] = []
        self.optimizer_iterations: int = 0
        self.final_calls: int = 0  # calls of ExecutionService.execute_final (also via execute_last_...)
        self.final_index: int | None = None  # index of the final circuit in ``circuits``

    def add(self, circuit: QuantumCircuit, shots: int, seconds: float, backend: BackendV2,  # noqa: ARG002
            layers: float = math.nan) -> None:
        """Record one executed circuit (``layers``: see :func:`layer_count`)."""
        start = time.perf_counter()
        device = metrics_stats(circuit)
        metrics_seconds = time.perf_counter() - start
        self.metrics_seconds += metrics_seconds
        ideal = circuit_stats(circuit)
        self.circuits.append({
            "depth": device["depth"],
            "two_qubit_gates": device["two_qubit_gates"],
            "total_gates": device["total_gates"],
            "physical_qubits": device["qubits"],
            "qpu_seconds": estimate_qpu_seconds(device["duration"], shots, get_backend(config.METRICS_BACKEND)),
            "ideal_depth": ideal["depth"],
            "ideal_two_qubit_gates": ideal["two_qubit_gates"],
            "ideal_total_gates": ideal["total_gates"],
            "num_qubits": ideal["qubits"],
            "layers": layers,
            "shots": shots,
            "execute": seconds,
            "metrics_seconds": metrics_seconds,
        })

    def elapsed(self) -> float:
        """Wall-clock seconds since the log was created, without the metrics transpilation."""
        return time.perf_counter() - self.start - self.metrics_seconds

    def remaining(self) -> float:
        """Seconds left until the time limit. Algorithms can use it to stop in time and return what they
        have, instead of breaking the limit (which ends the run without samples)."""
        _, time_limit = limits()
        return time_limit - self.elapsed()

    def check_time(self) -> None:
        """Fail the run once it took longer than the benchmark's time limit."""
        _, time_limit = limits()
        if self.elapsed() > time_limit:
            msg = f"Time limit of {time_limit:.0f} s exceeded after {len(self.circuits)} circuits"
            raise LimitExceeded(msg, self)

    def check_final(self, solution: Solution) -> None:
        """Fail the run unless its solution consists of the final circuit's shots: ``execute_last_and_return_solution``
        or ``execute_final`` called exactly once, no circuit executed after it, and exactly ``rules.FINAL_SHOTS``
        samples in ``solution``."""
        self.check_time()
        if self.final_calls != 1:
            msg = (f"execute_last_and_return_solution or execute_final must be called exactly once, "
                   f"it was called {self.final_calls} times")
            raise FinalShotViolation(msg, self)
        if self.final_index != len(self.circuits) - 1:
            msg = f"{len(self.circuits) - 1 - self.final_index} circuits were executed after the final circuit"
            raise FinalShotViolation(msg, self)
        samples = int(solution.counts.sum()) if len(solution.samples) else 0
        if samples != rules.FINAL_SHOTS:
            msg = f"The solution has {samples} samples, not the FINAL_SHOTS={rules.FINAL_SHOTS} of the final circuit"
            raise FinalShotViolation(msg, self)

    def timing(self) -> Timing:
        """All statistics as a luna Timing; ``total`` is :meth:`elapsed`."""
        entries = {key: [float(c[key]) for c in self.circuits] for key in self.circuits[0]} if self.circuits else {}
        entries["optimizer_iterations"] = float(self.optimizer_iterations)
        return Timing.from_dict(entries, total=self.elapsed())
