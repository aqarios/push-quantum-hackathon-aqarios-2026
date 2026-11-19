from typing import Literal

from luna_model import Model, Solution
from pydantic import PrivateAttr, Field
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from bench.building_blocks import cvar, expectation, ising_operator, lp_to_qubo, pair_rounds, scaled_ising
from bench.execution import HackathonParticipantAlgorithm, algorithm, ExecutionService
from .utils import apply_measurements, generate_initial_state, minimize

import numpy as np

@algorithm(algorithm_id="example.QAOA")
class TextBook_QAOA(HackathonParticipantAlgorithm):
    """Standard TextBook QAOA implementation (extend it for your ideas).
    Some ideas for basic improvements are marked as hints in comments here and in utils.py!
    If you are new to quantum computing or to QAOA in particular, read the comments in the functions: they explain the basic math and functionality.
    Feel free to extend this class with new arguments for different optimizers or create your own.

    Attributes
    ----------
    shots : int
        Number of samples. Adapt for your needs, change carefully, there is time restriction for whole computation
    optimizer : str
        Optimizer to use, by default COBYLA.
        But of course you can use everything what scipy_minimize supports.
        Or even implement your own. Consider minimize(...) in utils.py
    cost_function : str
        Which cost function evaluates the counts: the expectation value or the CVaR (conditional value at risk)
    penalty_scaling : float
        How much violation of constraint will cost for the function that we are trying to minimize.
        Example:
            f(x) is our cost function, that we want to minimize, let P(x) be a function that encodes our constraints. P(x) grows when violation is harder.
            Then objective function to minimize is C(x) = f(x) + penalty_scaling * P(x).
    max_iterations : int
        Iterations number optimizer allowed to run.
        Change free but carefully, there is time restriction for whole computation
    learning_rate : float
        Maybe useful for SPSA algorithm or another. COBYLA doesn't use it.
        Refer to specific algorithm
    reps : int
        Number of layers in QAOA
    layers_init_type : str
        Represent initial parameters filling strategy.
        ramp: uses Linear Ramp Initialization
        random: generates random floats
        self_passed: takes 'initial_gammas' and 'initial_betas', need to be the same length as 'reps'
    ramp_evolution_time : float
        'dt' parameter in Linear Ramp Initialization formula
    initial_gammas : list[float]
        An array of initial gamma parameters for cost layers used by optimizer.
        Length need to be equal to 'reps'
        'layers_init_type' need to be set to "self_passed"
    initial_betas : list[float]
        An array of initial beta parameters for mixer layer used by optimizer.
        Length needs to be equal to 'reps'
        'layers_init_type' need to be set to "self_passed"
    """
    _model = PrivateAttr()
    _execution_service = PrivateAttr()
    _qubo = PrivateAttr()
    _num_qubits = PrivateAttr()
    _ising_h = PrivateAttr()
    _ising_j = PrivateAttr()
    _ising_as_operator = PrivateAttr()

    shots: int = Field(default=1024, ge=1)

    optimizer: str = Field(default="COBYLA")
    cost_function: Literal["expectation", "cvar"] = "expectation"
    alpha: float = Field(default=0.1, gt=0.0, le=1.0)  # used only when cost_function="cvar"
    penalty_scaling: float = Field(default=10.0, allow_inf_nan=False)
    max_iterations: int = Field(default=100, ge = 1)
    learning_rate: float = Field(default=0.1, gt = 0, allow_inf_nan=False)

    reps : int = Field(default=5, ge = 1)
    layers_init_type: Literal["random", "ramp", "self_passed"] = "random"
    ramp_evolution_time : float = Field(default=0.5) # used only when layers_init_type="ramp"
    initial_gammas: list[float] = Field(default=[])
    initial_betas: list[float] = Field(default=[])

    _parametrized_circuit = PrivateAttr()
    _gammas_parameter_vector = PrivateAttr()
    _betas_parameter_vector = PrivateAttr()

    def _cost_layer(self, circuit: QuantumCircuit, h: np.ndarray, j: np.ndarray, gamma: float) -> None:
        """Append ``exp(-i gamma H)`` for the Ising ``H`` of ``h`` and ``J``; ``gamma`` may be a Parameter.

        The RZZ gates commute, so they are applied in rounds of disjoint pairs (:func:`pair_rounds`): one RZ layer
        plus as few RZZ layers as possible on an all-to-all device.

        Intuitively, cost layer encodes quality of solution in phases of state

        Parameters
        ----------
        circuit : QuantumCircuit
            circuit on which layer will be applied
        h, J : np.ndarray
            Used to transform QUBO into Ising Hamiltonian
            Calculated by provided functions in run_with_logging()
        gamma : float
            One gamma parameter to apply at this layer, shows the strength of phase shift
        """
        for a in np.flatnonzero(h):
            circuit.rz(2 * gamma * h[a], int(a))
        for pairs in pair_rounds([(int(a), int(b)) for a, b in zip(*np.nonzero(j), strict=True)]):
            for a, b in pairs:
                circuit.rzz(2 * gamma * j[a, b], a, b)

    def _mixer_layer(self, circuit: QuantumCircuit, beta: float) -> None:
        """
        Turns the phases set by the cost layer into a changed probability distribution
        Hint: as simple improvement read about XY-mixer, Pay attention to impact on initial state!

        Parameters
        ----------
        circuit : QuantumCircuit
            circuit on which layer will be applied
        beta : float
            One beta parameter to apply at this layer, shows the strength of probability shift
        """
        circuit.rx(2 * beta, circuit.qubits)

    def _apply_one_layer(self, circuit: QuantumCircuit, h: np.ndarray, j: np.ndarray, gamma: float, beta: float) -> None:
        self._cost_layer(circuit, h, j, gamma)
        self._mixer_layer(circuit, beta)

    def _apply_qaoa_layers(self, circuit: QuantumCircuit, h: np.ndarray, j: np.ndarray, gammas: list[float], betas: list[float]) -> None:

        for gamma, beta in zip(gammas, betas):
            self._apply_one_layer(circuit, h, j, gamma, beta)

    def _get_counts(self, parameters:np.ndarray) -> dict[str, int]:
        """
        Takes array with parameters.
        Bound parametrized transpiled circuit with these parameters and simulates execution.
        Parameters
        ----------
        parameters : list[float]
            list with gammas and betas from optimizer

        Returns
        -------
        Dictionary with execution results, that need to be evaluated
        """
        gammas = parameters[:len(parameters) // 2]
        betas = parameters[len(parameters) // 2:]

        values = {
            **dict(zip(self._gammas_parameter_vector, gammas)),
            **dict(zip(self._betas_parameter_vector, betas)),
        }

        if len(values.items()) != 2*self.reps:
            raise RuntimeError("Amount of parameters in circuit should be 2*self.reps")

        counts = self._execution_service.execute(self._parametrized_circuit, values, self.shots)

        return counts

    def _cost_function(self, parameters:np.ndarray) -> float:
        """

        Parameters
        ----------
        parameters
            An Array with gammas and betas from optimizer

        Returns
        -------
        "Energy" of solution distribution, depending on selected cost function.
        This Energy we are trying to minimize, since lowest energy means best solution.
        """
        counts = self._get_counts(parameters)

        if self.cost_function == "expectation":
            energy = expectation(self._qubo, counts)
        elif self.cost_function == "cvar":
            energy = cvar(self._qubo, counts, self.alpha)
        else:
            raise ValueError(f"Unknown cost function.\n Have you defined it in {__file__} in cost_function and _cost_function()?")

        return energy

    def _create_parametrized_circuit(self):
        self._gammas_parameter_vector = ParameterVector("gamma", self.reps)
        self._betas_parameter_vector = ParameterVector("beta", self.reps)

        circuit_parametrized = generate_initial_state(self._num_qubits, "uniform")

        self._apply_qaoa_layers(circuit_parametrized, self._ising_h, self._ising_j,
                                self._gammas_parameter_vector,
                                self._betas_parameter_vector)

        apply_measurements(circuit_parametrized)
        return circuit_parametrized

    def _initialize_start_parameters(self):
        if self.layers_init_type == "self_passed":
            if self.reps != len(self.initial_gammas) or self.reps != len(self.initial_betas):
                raise ValueError(f"reps parameter must match your passed start gammas and betas")
            x0 = np.array([*self.initial_gammas, *self.initial_betas])
        elif self.layers_init_type == "random" and (self.initial_betas == [] and self.initial_gammas == []):
            x0 = np.random.default_rng().uniform(-30, 30, size = 2*self.reps)
        elif self.layers_init_type == "ramp" and (self.initial_betas == [] and self.initial_gammas == []):
            x0 = np.array([
                *[self.ramp_evolution_time * (i / (self.reps + 1)) for i in range(1, self.reps + 1)],
                *[-self.ramp_evolution_time * (1 - (i / (self.reps + 1))) for i in range(1, self.reps + 1)]
            ])
        else:
            raise ValueError(f"Unknown QAOA start parameters initialization.\n Have you defined it in {__file__} in layers_init_type and run()?")

        return x0

    def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
        """Creates basic QAOA instance and samples it"""
        self._model = model
        self._execution_service = execution_service

        self._qubo, transformation = lp_to_qubo(model, self.penalty_scaling)
        self._num_qubits = len(self._qubo)  # one qubit per QUBO variable
        self._ising_h, self._ising_j = scaled_ising(self._qubo)

        #probably is not needed, since we need to construct a time evolution,
        # and time evolution is done in a cost layer
        # but let it be here for observability
        self._ising_as_operator = ising_operator(self._ising_h, self._ising_j)

        x0 = self._initialize_start_parameters()

        self._parametrized_circuit = self._create_parametrized_circuit()

        optimized_parameters, iterations = minimize(
            cost=self._cost_function,
            x0=x0,
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            learning_rate=self.learning_rate,
            time_left=self._execution_service.remaining_time
            )

        # Need to be careful with parameter order, e.g. without this explicit mapping with ParameterVector it could lead to wrong ordering because of lexicographic sort from Qiskit side
        final_gammas = optimized_parameters[:len(optimized_parameters) // 2]
        final_betas = optimized_parameters[len(optimized_parameters) // 2:]
        final_parameters_mapping = {
            **dict(zip(self._gammas_parameter_vector, final_gammas)),
            **dict(zip(self._betas_parameter_vector, final_betas)),
        }
        solution = execution_service.execute_last_and_return_solution(transformation, model, self._parametrized_circuit, iterations, final_parameters_mapping)
        return solution
