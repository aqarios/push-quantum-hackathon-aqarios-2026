"""Building blocks of this team's algorithms that you will likely adapt. Your team folder has its own copy, so
change and extend them freely.

* :func:`generate_initial_state`: the state to start from; only the uniform superposition so far.
* :func:`apply_measurements`: measures every qubit at the end of a circuit.
* :func:`minimize`: an optimizer loop that stops in time (``time_left``).

The fixed building blocks (QUBO, cost functions, Ising form) are in ``bench.building_blocks``. Circuits are executed
with ``bench.execution.ExecutionService.execute`` (fixed: noiseless simulation, run log, limits).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize as scipy_minimize

if TYPE_CHECKING:
    from collections.abc import Callable


# --------------------------------------------------------------------------------------
# Circuits and optimization
# --------------------------------------------------------------------------------------


def generate_initial_state(num_qubits: int, kind: str = "uniform") -> QuantumCircuit:
    """State preparation from ``|0...0>``. Only the uniform superposition exists for now."""
    circuit = QuantumCircuit(num_qubits)
    if kind == "uniform":
        circuit.h(range(num_qubits))
        return circuit
    msg = f"Initial state {kind!r} is not implemented"
    raise NotImplementedError(msg)

def apply_measurements(circuit: QuantumCircuit):
    # creates new register
    circuit.measure_all()


class OutOfTime(Exception):  # noqa: N818 - reads better than OutOfTimeError
    """Raised inside :func:`minimize` when the time budget is used up."""


def minimize(  # noqa: PLR0913
    cost: Callable[[np.ndarray], float],
    x0: np.ndarray,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    seed: int | None = None,
    time_left: Callable[[], float] | None = None,
) -> tuple[np.ndarray, int]:
    """Minimize ``cost``; returns the best parameters and the number of iterations.

    ``optimizer`` is ``"SPSA"`` (uses ``learning_rate``) or a gradient-free method of
    ``scipy.optimize.minimize`` such as ``"COBYLA"``, ``"Nelder-Mead"`` or ``"Powell"``. With ``time_left``
    (seconds left: pass ``execution_service.remaining_time``) the optimization stops early with the best parameters
    so far once less than twice the longest cost evaluation is left, so the final sampling and ``backtransform``
    still fit into the time limit (a run that breaks it ends without samples).
    """
    best = {"x": np.asarray(x0, dtype=float), "cost": np.inf, "calls": 0, "longest": 0.0}

    def tracked_cost(x: np.ndarray) -> float:
        if time_left is not None:
            left = time_left()
            if left < 2 * best["longest"]:
                raise OutOfTime
        value = cost(x)
        if time_left is not None:  # measured with time_left, so time that does not count is left out
            best["longest"] = max(best["longest"], left - time_left())
        best["calls"] += 1
        if value < best["cost"]:
            best["x"], best["cost"] = np.array(x, dtype=float), value
        return value

    try:
        if optimizer.upper() != "SPSA":
            result = scipy_minimize(tracked_cost, x0, method=optimizer, options={"maxiter": max_iterations})
            return best["x"], int(getattr(result, "nit", None) or result.nfev)

        # SPSA (Spall 1998): two cost evaluations per iteration along a random direction.
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, dtype=float)
        for k in range(max_iterations):
            step, shift = learning_rate / (k + 1) ** 0.602, 0.1 / (k + 1) ** 0.101
            delta = rng.choice([-1.0, 1.0], size=x.shape)
            plus, minus = tracked_cost(x + shift * delta), tracked_cost(x - shift * delta)
            x = x - step * (plus - minus) / (2 * shift) * delta
        return best["x"], max_iterations
    except OutOfTime:
        return best["x"], best["calls"]
