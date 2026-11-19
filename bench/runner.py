"""Command line and luna_bench run behind run_benchmark.py: instances by size class, results and figures.

Results (results.csv) and figures go to reports/NN_benchmark-<size> (NN: the number of the size class, so the
folders sort from small to very large) or to reports/<name> with --name; other files there are kept. Rules:
bench/rules.py; metrics and figures: bench/config.py.

luna_bench stores everything under ./benchmark_data. Re-running keeps the results of unchanged algorithms and
runs only new or changed ones (removing an algorithm removes its results; --only runs a subset now and leaves the
others as they are; --rerun runs unchanged ones again); features, metrics and figures are recomputed every time.
Failed or interrupted runs are repeated every time, also those that fail again for sure (e.g. too many qubits);
--no-retry keeps them as failed; --clean deletes the folder first and starts from scratch. After aborting a run,
delete benchmark_data/database/luna_bench-jobs.db: luna_bench keeps queued jobs and the next run executes them.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from luna_bench import Benchmark
from luna_bench.configs.config import config as luna_bench_config
from luna_bench.errors.registry.unknown_id_error import UnknownIdError

from bench import config, rules
from bench.figures import draw_figures, tts_ranking
from bench.utils import (
    INSTANCES_FILE,
    add_algorithms,
    add_features,
    add_metrics,
    build_modelset,
    load_instances,
    qubo_size,
    record_luna_shots,
    size_class,
)

record_luna_shots()  # FlexQAOA's shots (from the solve job) for the shots-to-solution metric


def run(algorithms: dict[str, Any], description: str) -> None:
    """Benchmark ``algorithms`` (name in results and figures -> algorithm) on the instances up to ``--size``."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=list(rules.SIZES), default="small",
                        help="instances up to this size class (default: small); "
                             + ", ".join(f"{name}: <= {largest} QUBO variables" for name, largest in rules.SIZES.items()))
    parser.add_argument("--instances", type=Path, default=INSTANCES_FILE, metavar="FILE",
                        help="instance file in the format of bench/data/instances.yaml, e.g. a grading set kept out "
                             "of the repo; its name is appended to the default name (default: the 47 instances)")
    parser.add_argument("--tag", default=None,
                        help="only instances with this tag in the instance file; the tag is appended to the "
                             "default name (default: all instances)")
    parser.add_argument("--name", default=None,
                        help="benchmark name; results are stored per name, the report goes to reports/<name> "
                             "(default: benchmark-<size>[-<tag>][-<instance file>], report in "
                             "reports/NN_benchmark-<size>[-<tag>][-<instance file>])")
    parser.add_argument("--only", nargs="+", default=None, metavar="ALGORITHM",
                        help="run only these algorithms, or all algorithms of these teams, now; the others keep their "
                             "stored results (or are not run yet) (default: all algorithms)")
    parser.add_argument("--rerun", nargs="+", default=[], metavar="ALGORITHM",
                        help="run these algorithms, or all algorithms of these teams, again although their parameters "
                             "did not change (default: none; an algorithm runs only if it is new or its parameters "
                             "changed)")
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True,
                        help="run failed or interrupted runs again (default: yes; --no-retry keeps them as failed)")
    parser.add_argument("--time-limit", type=float, default=rules.TIME_LIMIT,
                        help=f"seconds per run of a local algorithm (default: {rules.TIME_LIMIT:g})")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the transpiler, the simulator and the metrics transpilation (default: random)")
    parser.add_argument("--list", action="store_true", help="list the instances with their size class and exit")
    parser.add_argument("--list-algorithms", action="store_true", help="list the discovered algorithms to run and exit")
    parser.add_argument("--clean", dest="clean", action="store_true",
                        help="delete luna_bench's data folder (./benchmark_data, or LB_DATA_DIR) before the run, so "
                             "everything is computed from scratch; this also deletes the stored results of every other "
                             "algorithm and benchmark (default: keep it)")

    args = parser.parse_args()
    if args.list:
        for problem, model in load_instances(args.tag, args.instances):
            size = qubo_size(model)
            print(f"{problem['name']:16s} {size:3d} QUBO variables  {size_class(size)}")
        return

    if args.list_algorithms:
        print(f"Discovered Algorithms: {list(algorithms)}")
        return

    if args.clean:
        # The folder luna_bench itself uses: LB_DATA_DIR, default ./benchmark_data relative to the current directory.
        data_dir = Path(luna_bench_config.LB_DATA_DIR).expanduser().resolve()
        shutil.rmtree(data_dir, ignore_errors=True)
        print(f"Deleted luna_bench's data folder {data_dir}")

    try:
        only = None if args.only is None else select(args.only, algorithms)
        rerun = select(args.rerun, algorithms)
    except KeyError as error:
        parser.error(f"unknown algorithm or team {error.args[0]!r}; choose from {list(algorithms)}")
    name = args.name or (f"benchmark-{args.size}" + (f"-{args.tag}" if args.tag else "")
                         + ("" if args.instances.resolve() == INSTANCES_FILE else f"-{args.instances.stem}"))
    os.environ["BENCH_TIME_LIMIT"] = str(args.time_limit)  # read by bench.execution, also in the worker
    if args.seed is not None:
        os.environ["BENCH_SEED"] = str(args.seed)  # read by bench.config in the worker, which runs the circuits

    try:
        bench = Benchmark.create(name)
    except UnknownIdError as error:  # luna_bench loads every stored component by its registered id
        parser.error(f"the stored benchmark {name!r} holds {error.unknown_id!r} ({error.registry}), which no code "
                     "registers any more, e.g. after a renamed algorithm_id or a removed team folder, so luna_bench "
                     "cannot load it. Run it under a new --name, or move benchmark_data/ aside to start every "
                     "benchmark fresh.")
    bench.set_modelset(build_modelset(f"{name}-models", args.size, args.tag, args.instances))
    add_algorithms(bench, algorithms, rerun=tuple(rerun), only=only)
    add_features(bench, config.FEATURES)
    add_metrics(bench, config.METRICS)
    bench.run(retry_uncompleted=args.retry)
    folder = name if args.name else f"{list(rules.SIZES).index(args.size) + 1:02d}_{name}"
    print(f"Results and figures: {write_report(bench, Path('reports') / folder)}")


def select(names: list[str], algorithms: dict[str, Any]) -> list[str]:
    """``names`` with every team name replaced by that team's algorithms (``<team>-*``); raises ``KeyError`` for a
    name that is neither an algorithm nor a team."""
    selected = []
    for name in names:
        matches = [name] if name in algorithms else [other for other in algorithms if other.startswith(f"{name}-")]
        if not matches:
            raise KeyError(name)
        selected += [match for match in matches if match not in selected]
    return selected


def write_report(bench: Benchmark, report_dir: Path) -> Path:
    """Figures, the QPU-TTS ranking and results.csv of ``bench`` in ``report_dir``; other files there are kept."""
    report_dir.mkdir(parents=True, exist_ok=True)
    for old in [*report_dir.glob("*.png"), report_dir / "results.csv"]:
        old.unlink(missing_ok=True)
    draw_figures(bench, config.FIGURES, report_dir)
    frame = bench.to_dataframe()
    frame.to_csv(report_dir / "results.csv", index=False)
    (report_dir / "qpu_tts_ranking.png").unlink(missing_ok=True)  # older reports
    tts_ranking(frame, report_dir / "tts_ranking.png")
    return report_dir
