"""Solve a single instance with one algorithm and print the solution and the benchmark metrics.

    uv run python examples/solve_one_problem.py                            # example-MyAlgorithm on traffic_c3
    uv run python examples/solve_one_problem.py maxcut_n10 example-QAOA    # any instance and team algorithm

The quickest loop for trying out an idea: no benchmark database, but the same features (optimum by SCIP) and
metrics as the benchmark (bench/config.py).
"""

import argparse

from luna_bench.custom import FeatureResultContainer

from bench.config import FEATURES, METRICS
from bench.registry import discover
from bench.utils import load_instances

# Every team's algorithms, the same as in run_benchmark.py: ALGORITHMS of hackathon/teams/<team>/__init__.py.
ALGORITHMS = discover()


def main() -> None:
    """Solve the instance and print solution and metrics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance", nargs="?", default="traffic_c3", help="name in bench/data/instances.yaml")
    parser.add_argument("algorithm", nargs="?", default="example-MyAlgorithm", choices=list(ALGORITHMS))
    args = parser.parse_args()

    model = {problem["name"]: model for problem, model in load_instances()}[args.instance]
    print(model)
    solution = ALGORITHMS[args.algorithm].run(model)
    print(solution)

    features = FeatureResultContainer.model_construct(
        data={type(feature): {name: (feature.run(model), feature)} for name, feature in FEATURES.items()})
    for name, metric in METRICS.items():
        print(f"{name}: {metric.run(solution, features)}")


if __name__ == "__main__":
    main()
