"""Customizable variational quantum Eigensolver (VQE) for constrained optimization.

This example converts a constrained model into a QUBO, trains a parameterized
circuit using measured samples, and maps the final samples back to the
original model.

Available ansatz examples:
- hardware_efficient: Ry-Rz rotations with full CX entanglement
- real_amplitudes: Ry rotations with sparse neighbouring CX gates

Available losses:
- expectation: mean sampled QUBO energy.
- cvar: mean energy of the best alpha fraction of samples
  (Barkoutsos et al., Quantum 4, 256 (2020))

To experiment with your own algorithm:
- Modify the ansatz based on your circuit design
- Implement or modify the cost function. Here we provide two examples: QUBO Energy and CVaR,
  you can design your own or implement one from a paper, for example the tilted loss of arXiv:2605.02850
- Experiment with different optimizers and hyperparameters
- Extend the implementation with more advanced approaches, such as constraint-preserving VQE, ADAPT-VQE,
  layerwise VQE, and more.

All training and final-sampling circuits run through
``bench.execution.ExecutionService.execute`` to enforce benchmark limits and record metrics.
"""

import numpy as np
from typing import Literal
from collections.abc import Mapping
from pydantic import Field

from luna_model import Model, Solution
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from bench.execution import HackathonParticipantAlgorithm, algorithm, ExecutionService
from bench.building_blocks import lp_to_qubo, expectation, cvar
from .utils import minimize

@algorithm(algorithm_id="example.CustomVQE")
class CustomVQE(HackathonParticipantAlgorithm):
    """Custom VQE for QUBO optimization with independently selectable ansatz and training loss.

    For hardware_efficient and real_amplitudes, reps counts CX blocks.
    Constraints are handled through QUBO penalties, not enforced directly
    by the circuits.

    For parameters specific to custom loss function: 'alpha' controls CVaR.
    """

    seed: int = 42
    shots: int = Field(default=1024, ge=1)
    max_iterations: int = Field(default=200, ge=1)
    optimizer: str = "COBYLA"
    learning_rate: float = Field(default=0.1, gt=0.0)
    penalty_scaling: float = 10.0

    # Ansatz
    reps: int = Field(default=2, ge=0)
    ansatz_type: Literal["hardware_efficient", "real_amplitudes"] = "hardware_efficient"

    # Cost Function
    cost_function: Literal["expectation", "cvar"] = "expectation"
    alpha: float = Field(default=0.1, gt=0.0, le=1.0)  # used only for cost function = CVaR

    def _construct_ansatz(self, qubo: np.ndarray) -> tuple[QuantumCircuit, ParameterVector]:
        """Return the selected unmeasured circuit and ordered parameters.
        To add an ansatz: Implement a circuit-building helper function, then add its name to ansatz_type,
        and dispatch to the helper here.
        """
        num_qubits = len(qubo)

        if self.ansatz_type == "hardware_efficient":
            return self._construct_hardware_efficient(num_qubits, self.reps)
        if self.ansatz_type == "real_amplitudes":
            return self._construct_real_amplitudes(num_qubits, self.reps)

        raise ValueError(f"unknown ansatz type: {self.ansatz_type}")

    @staticmethod
    def _construct_hardware_efficient(num_qubits: int, reps:int = 2) -> tuple[QuantumCircuit, ParameterVector]:
        """Build an Ry-Rz ansatz with full CX entanglement adapted from qiskit.circuit.library.efficient_su2
        Uses ``reps + 1`` rotation layers separated by ``reps`` entangling
        blocks, with CX gates on every pair ``i < j``.
        """

        num_params = 2 * num_qubits * (reps + 1)
        theta = ParameterVector("theta", num_params)
        circuit = QuantumCircuit(num_qubits)

        k = 0
        for layer in range(reps + 1):
            for qubit in range(num_qubits):
                circuit.ry(theta[k], qubit)  # parameterized rotation layer
                k += 1
                circuit.rz(theta[k], qubit)
                k += 1

            if layer < reps:
                for control in range(num_qubits):
                    for target in range(control + 1, num_qubits):
                        circuit.cx(control, target)

        return circuit, theta

    @staticmethod
    def _construct_real_amplitudes(num_qubits: int, reps: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
        """Build Ry layers separated by sparse CX blocks.
        Each CX block has two rounds of disjoint neighbouring pairs.
        With reps=0, the circuit is a product state.
        """

        num_parameters = num_qubits * (reps + 1)
        parameters = ParameterVector("theta", num_parameters)
        circuit = QuantumCircuit(num_qubits)

        parameter_index = 0

        for layer in range(reps + 1):
            for qubit in range(num_qubits):
                circuit.ry(parameters[parameter_index], qubit)
                parameter_index += 1

            if layer < reps:
                for parity in (0, 1):
                    for control in range(parity, num_qubits - 1, 2):
                        circuit.cx(control, control + 1)

        return circuit, parameters

    def _evaluate_cost(self, qubo: np.ndarray, counts: Mapping[str, int]) -> float:
        """Evaluate the selected loss using measured QUBO samples."""
        if self.cost_function == "expectation":
            return expectation(qubo, counts)
        if self.cost_function == "cvar":
            return cvar(qubo, counts, alpha=self.alpha)
        raise ValueError(f"Unknown cost function: {self.cost_function}")

    def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
        """Train the selected circuit and loss function, then return its final samples."""
        qubo, transformation = lp_to_qubo(model, self.penalty_scaling)

        # Construct the selected parameterized circuit (ansatz)
        ansatz, theta = self._construct_ansatz(qubo)
        ansatz.measure_all()  # measure all qubits

        # Define cost function (minimization objective)
        def objective(params: np.ndarray) -> float:
            values = {}
            for parameter, value in zip(theta, params):
                values[parameter] = value

            counts = execution_service.execute(ansatz, values, shots=self.shots) # bound values and executes
            return self._evaluate_cost(qubo, counts)

        # Classical optimization
        rng = np.random.default_rng(self.seed) # Numpy random number generator for parameter initialization
        x0 = rng.uniform(0, 2 * np.pi, len(theta))
        best_params, iterations = minimize(
            cost=objective,
            x0=x0,
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            learning_rate=self.learning_rate,
            seed=self.seed,
            time_left=execution_service.remaining_time,
        )
        # Sample the optimized circuit
        solution = execution_service.execute_last_and_return_solution(transformation, model, ansatz, iterations,
                                                                      best_params)
        return solution