"""Benchmark rules, fixed for all participants: the qubit and time limit of a run, and the size classes of the
instances.

``bench.execution`` enforces both limits for every algorithm: a run that breaks one ends with an empty solution and
counts as not solved.

This module imports nothing, so every module of ``bench`` can import it at the top without an import cycle.
"""

# Logical qubits per circuit, ancillas included. 25 qubits fit onto the 27 of METRICS_BACKEND (bench/config.py); the
# largest instances have 24 QUBO variables.
MAX_QUBITS = 25

# Seconds per run, checked before and after every circuit (a running circuit is not interrupted). The metrics
# transpilation onto METRICS_BACKEND does not count.
TIME_LIMIT = 2 * 60

# Shots of the final circuit, the one whose samples are the solution: exactly this many, executed once and last with
# ``ExecutionService.execute_last_and_return_solution`` (or ``execute_final`` for an encoding of your own). The
# optimization before it may use any number of shots within the time limit.
FINAL_SHOTS = 1024

# Algorithms per team in the ranking: the first ones of ALGORITHMS by name. The example team is exempt.
MAX_ALGORITHMS_PER_TEAM = 2

# Size classes of the instances by QUBO variables (slack variables included): the largest size of each class.
# ``--size mid`` runs the small and the mid instances; start small, then go up step by step.
SIZES = {"small": 8, "mid": 16, "large": 20, "very-large": 24}
