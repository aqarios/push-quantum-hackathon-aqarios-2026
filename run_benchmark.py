"""Benchmark the Luna baselines of ALGORITHMS and every team's algorithms (hackathon/teams/) on the instances of
bench/data/instances.yaml.

Usage::

    uv run python run_benchmark.py --list               # the instances and their size class
    uv run python run_benchmark.py                      # small instances (<= 8 QUBO variables) -> reports/01_benchmark-small
    uv run python run_benchmark.py --size mid           # small and mid instances (<= 16)
    uv run python run_benchmark.py --size large         # ... up to 20
    uv run python run_benchmark.py --size very-large    # all instances (<= 24)
    uv run python run_benchmark.py --only example-QAOA  # run only this algorithm now, keep the results of the others
    uv run python run_benchmark.py --only alpha         # run all algorithms of team alpha now
    uv run python run_benchmark.py --clean              # delete all stored results first, compute everything again
    uv run python run_benchmark.py --size mid --instances grading.yaml --seed 7   # another instance file, fixed seeds

Start small: a run of every algorithm on the small instances takes a few minutes, the very large instances
can take up to the time limit (2 min) per algorithm and instance. FlexQAOA, QAOA-Luna and VQE-Luna run on the
Luna platform and need LUNA_API_KEY in .env; comment them out to work offline. Command line and storage:
bench/runner.py.
"""

from luna_quantum.solve.parameters.algorithms import QAOA, VQE, FlexQAOA
from luna_quantum.solve.parameters.algorithms.base_params.qaoa_circuit_params import LinearQAOAParams
from luna_quantum.solve.parameters.algorithms.base_params.scipy_optimizer import ScipyOptimizerParams
from luna_quantum.solve.parameters.algorithms.quantum_gate.flexqaoa import (
    IndicatorFunctionConfig,
    PenaltySetting,
    PipelineParams,
    QuadraticPenaltyConfig,
)
from luna_quantum.solve.parameters.backends import Aqarios, AqariosGpu  # AqariosGpu: the GPU variant below
from luna_quantum.solve.parameters.backends.ibm.simulator import AerSimulator

from bench.registry import discover
from bench.runner import run

# Name in results and figures -> algorithm. Local algorithms run on the ideal backend (noiseless, all-to-all).
# The others run on the Luna platform's noiseless CPU simulators, both limited:
# - FlexQAOA: state vectors up to 2^18, but FlexQAOA only simulates the subspace its constraints allow (one-hot
#   via XY mixers), so most instances fit; not the ones without one-hot constraints above 18 variables
#   (maxcut_l20, partition_l20, partition_x24: the job fails).
# - QAOA-Luna, VQE-Luna (Luna's QAOA and VQE on the model's QUBO, Qiskit Aer): at most 18 qubits, i.e. QUBO
#   variables; the 7 instances above (4 large, 3 very large) fail. A local implementation has no such limit.
ALGORITHMS = {
    "QAOA-Luna": QAOA(
        reps=5,
        optimizer=ScipyOptimizerParams(method="cobyla", maxiter=200),
        backend=AerSimulator()
    ),
    "VQE-Luna": VQE(  # hardware-efficient ansatz (EfficientSU2), full entanglement
        ansatz="efficient_su2",
        ansatz_config={"reps": 2, "entanglement": "full"},
        optimizer=ScipyOptimizerParams(method="cobyla", maxiter=100),
        backend=AerSimulator(),
    ),
    "FlexQAOA": FlexQAOA(  # tuned parameters
        reps=360,
        optimizer=ScipyOptimizerParams(method="cobyla", maxiter=100),
        param_conversion=None,  # only the two slopes of the linear schedule
        initial_params=LinearQAOAParams(delta_beta=0.25, delta_gamma=0.25),
        pipeline=PipelineParams(
            penalty=PenaltySetting(scaling=2.0),
            indicator_function=IndicatorFunctionConfig(penalty=PenaltySetting(scaling=0.5)),
            quadratic_penalty=QuadraticPenaltyConfig(penalty=PenaltySetting(scaling=0.25)),
            sp_quadratic_penalty=QuadraticPenaltyConfig(penalty=PenaltySetting(scaling=0.5)),
        ),
        backend=Aqarios(),
    ),
    # FlexQAOA on the GPU simulator (every instance fits), same tuned parameters:
    # "FlexQAOA-GPU": FlexQAOA(
    #     reps=360,
    #     optimizer=ScipyOptimizerParams(method="cobyla", maxiter=100),
    #     param_conversion=None,
    #     initial_params=LinearQAOAParams(delta_beta=0.25, delta_gamma=0.25),
    #     pipeline=PipelineParams(
    #         penalty=PenaltySetting(scaling=2.0),
    #         indicator_function=IndicatorFunctionConfig(penalty=PenaltySetting(scaling=0.5)),
    #         quadratic_penalty=QuadraticPenaltyConfig(penalty=PenaltySetting(scaling=0.25)),
    #         sp_quadratic_penalty=QuadraticPenaltyConfig(penalty=PenaltySetting(scaling=0.5)),
    #     ),
    #     backend=AqariosGpu(),
    # ),
}

# Every team's algorithms, from ALGORITHMS in hackathon/teams/<team>/__init__.py, named <team>-<name>.
ALGORITHMS |= discover()

if __name__ == "__main__":
    run(ALGORITHMS, description=__doc__)
