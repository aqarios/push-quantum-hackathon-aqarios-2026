"""Benchmark infrastructure: rules, execution, evaluation and plots. Fixed for all participants."""

import os

from dotenv import load_dotenv

load_dotenv()  # LUNA_API_KEY for FlexQAOA

# luna_bench reads these when it is first imported (bench.execution imports it, so they are always set first).
# One run at a time: parallel simulations distort runtimes and time limits and add up in RAM.
os.environ.setdefault("LB_ASYNC_WORKER_COUNT", "1")
# luna_bench polls for finished runs every millisecond with a write transaction on its SQLite job database;
# while a simulation kept all cores busy this ended in "database is locked". Every 0.2 s is plenty.
os.environ.setdefault("LB_ALGORITHM_INTERNAL_BACKOFF_TIME", "0.2")

# Every solution is evaluated on the variables of the original model only: helper variables an algorithm returns
# (ancillas of its own encoding, FlexQAOA's dummy_packing_element_* for set packing) are dropped first.
#
# This is the ONLY place that installs this filter, and it must stay here. The benchmark runner, the example, the
# tests and luna_bench's worker all get it from this import; none of them calls it on its own.
#
# Why it must be here: luna_bench runs every local algorithm in a separate worker process. That process never
# runs the runner, the example or the tests. It only imports the algorithm's module, which imports bench, so this
# file is the only code of ours that runs there. Without the filter, an algorithm that returns a helper variable
# works in examples/solve_one_problem.py but fails in the benchmark with "variable does not exist".
#
# Why last: bench.utils imports luna_bench, which reads the LB_* variables above when it is first imported.
from bench.utils import evaluate_model_variables_only  # noqa: E402

evaluate_model_variables_only()
