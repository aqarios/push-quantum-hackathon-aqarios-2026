"""The example team: the template ``MyAlgorithm`` (random sampling), a textbook QAOA and a VQE, as reference
algorithms.

Start your team by copying this folder to ``hackathon/teams/<team>/``. ``ALGORITHMS`` lists what the benchmark runs;
results and figures name each entry ``<team>-<name>``, here ``example-MyAlgorithm``, ``example-QAOA`` and the two
``example-VQE-HE-...``. Every class needs an ``algorithm_id`` that starts with ``"<team>."``.
"""

from .my_algorithm import MyAlgorithm
from .qaoa import TextBook_QAOA
from .custom_vqe import CustomVQE

ALGORITHMS = {
    "MyAlgorithm": MyAlgorithm(),
    "QAOA": TextBook_QAOA(
        shots=1024,
        optimizer="COBYLA",
        cost_function="expectation",
        alpha=0.4,
        penalty_scaling=8.0,
        max_iterations=200,
        learning_rate=0.1,
        reps=5,
        layers_init_type="ramp",
        initial_gammas=[],
        initial_betas=[],
    ),
    # Custom VQE Experiment: Compare losses using the same hardware-efficient ansatz.
    "VQE-HE-Expectation": CustomVQE(
        ansatz_type="hardware_efficient",
        reps=2,
        cost_function="expectation",
    ),
    "VQE-HE-CVaR": CustomVQE(
        ansatz_type="hardware_efficient",
        reps=2,
        cost_function="cvar",
        alpha=0.25,
    ),
}
