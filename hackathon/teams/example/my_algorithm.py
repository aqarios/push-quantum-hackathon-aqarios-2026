"""Template for your own algorithm: rename the class, set its ``algorithm_id`` to ``"<team>.<Name>"``, fill in
``run_with_logging``.

An algorithm is a pydantic model (its fields are the parameters, shown in the results) with one method,
``run_with_logging(self, model, execution_service) -> Solution``. Inside ``run_with_logging``:

1. turn the constrained model into a QUBO (``lp_to_qubo``; or work with the model's constraints directly),
2. build circuits and execute them with ``execution_service.execute``: it transpiles them for the simulated
   backend, runs them without noise, records every circuit for the metrics and enforces the rules (at most
   MAX_QUBITS qubits, TIME_LIMIT seconds per run),
3. execute the circuit you want as the solution last, and return its samples mapped back to the model
   (``execution_service.execute_last_and_return_solution``).

The example below is the simplest possible "quantum" algorithm, sampling the uniform superposition (random
guessing), so that you have something running end to end; its comments walk through the steps. List your
algorithm in ``ALGORITHMS`` of your team's ``__init__.py``: ``run_benchmark.py`` and ``examples/solve_one_problem.py``
find it there.
"""

from luna_model import Model, Solution
from qiskit import QuantumCircuit

from bench.building_blocks import lp_to_qubo
from bench.execution import ExecutionService, HackathonParticipantAlgorithm, algorithm


@algorithm(algorithm_id="example.MyAlgorithm")
class MyAlgorithm(HackathonParticipantAlgorithm):
    """Uniform random sampling (replace with your idea).

    Attributes
    ----------
    shots : int
        Number of samples.
    penalty_scaling : float
        Penalty factor for turning constraints into QUBO penalties.
    """

    shots: int = 1024
    penalty_scaling: float = 10.0

    def run_with_logging(self, model: Model, execution_service: ExecutionService) -> Solution:
        # 1. QUBO: one qubit per QUBO variable.
        qubo, transformation = lp_to_qubo(model, self.penalty_scaling)
        n = len(qubo)

        # 2. Circuit: here the uniform superposition. A parametrized circuit (QAOA, VQE) gets its values in execute().
        circuit = QuantumCircuit(n, n)
        circuit.h(range(n))
        circuit.measure(range(n), range(n))

        # 3. Execute, e.g. inside an optimizer loop. execute() returns the counts {"0101...": count, ...}; a cost
        #    function turns them into one number to minimize, e.g. expectation(qubo, counts) or
        #    cvar(qubo, counts, alpha) from bench/building_blocks.py.
        counts = execution_service.execute(circuit, None, shots=self.shots)

        # 4. Time: everything in run_with_logging counts toward the time limit, and a run over it returns no samples.
        #    Check the seconds left to stop in time with the best result so far:
        #        seconds_left = execution_service.remaining_time()
        #    minimize in utils.py of this folder does this for you. Pass the method itself, not its value, and report the
        #    optimizer's iterations for the results:
        #        x, iterations = minimize(cost, x0, time_left=execution_service.remaining_time)
        #        execution_service.record_optimizer_iterations(iterations)

        # 5. Solution: it comes from one counts dict only. If you execute several circuits (see the QAOA and VQE
        #    examples), decide which one you want as the solution. The layer TTS metric (circuit layers x shots
        #    needed to sample the optimum with 99 % probability) uses the layers of the last executed circuit, so
        #    execute the circuit you think is best last. execute_last_and_return_solution executes it and maps its
        #    samples back to the model (backtransform).
        #    Hint: you don't know the optimum during the run, so compare cost functions to find the best circuit,
        #    e.g. expectation and cvar in bench/building_blocks.py, or write your own. Different cost functions can do
        #    better on different metrics.
        last_circuit = circuit.copy()
        solution = execution_service.execute_last_and_return_solution(transformation, model, last_circuit, 0, None)
        return solution
