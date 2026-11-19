# PushQuantum Hackathon 2026

Build a gate-based quantum algorithm that solves constrained optimization problems as well as possible, and
compare it with the baselines on the Luna platform (Aqarios): FlexQAOA, and Luna's standard QAOA and VQE
(`QAOA-Luna`, `VQE-Luna`), with random sampling (the template `example-MyAlgorithm`), a textbook QAOA
(`example-QAOA`) and a VQE with two cost functions (`example-VQE-HE-Expectation`, `example-VQE-HE-CVaR`).
Everything runs on a noiseless simulator; the benchmark measures the quality of your samples and estimates what
your circuits would cost on a real quantum computer.

The platform baselines run on Luna's CPU simulators, which have limits: `QAOA-Luna` and `VQE-Luna` take at most
18 qubits (they fail on the 7 instances with more QUBO variables), FlexQAOA fails on 3 instances whose state
vector exceeds 2^18 (see `run_benchmark.py`). Your local algorithms are only bound by the rules below.

## Setup

1. Register on [app.aqarios.com](https://app.aqarios.com) (one account per person).
2. Install [uv](https://docs.astral.sh/uv/) (Python package manager).
3. In this folder: `uv sync` (creates `.venv` with Python 3.13+ and all packages, every team's included).
4. For the platform baselines: copy `.env.example` to `.env` and put in your Luna API key (each person gets
   one). In `run_benchmark.py` you can configure the existing cloud algorithms, add others from the Luna
   platform, or disable some by commenting them out. Cloud compute is limited: copy the corresponding folder
   in `reports/` to keep the results of cloud jobs, since `--clean` deletes them. Without a key, comment the
   cloud algorithms out in `run_benchmark.py`; everything else works offline.
5. Check: `uv run pytest` (a minute or two).

## Quick start

```bash
uv run python run_benchmark.py --list                     # the 47 instances and their size class
uv run python run_benchmark.py --list-algorithms          # prints all registered algorithms
uv run python examples/solve_one_problem.py               # example-MyAlgorithm on one instance: solution and metrics
uv run python examples/solve_one_problem.py maxcut_n10 example-QAOA
uv run python run_benchmark.py                            # all algorithms on the small instances
uv run python run_benchmark.py --size mid                 # small and mid instances; then go up step by step: large, very-large
uv run python run_benchmark.py --only example             # only the algorithms of team example
uv run python run_benchmark.py --clean                    # delete all stored results first, compute everything again
```

Results and figures land in `reports/NN_benchmark-<size>/` (01 small ... 04 very-large): `results.csv` with
every metric per algorithm and instance, one PNG per metric, and `tts_ranking.png`, the algorithms sorted by
unsolved instances, then by time to solution in circuit layers.

## Instances and size classes

`bench/data/instances.yaml` lists 47 instances of 9 problem classes, generated with `luna_usecases`: MaxCut
(no constraints), traffic flow (one-hot), quadratic assignment (two-way one-hot / permutation), container
assignment (capacity inequalities), set packing (`sum <= 1`), production assignment and scheduling (mixed),
graph partitioning (`sum = n/2`), minimum vertex cover (`sum >= 1`) and facility location (one-hot +
implications). Each comes as a constrained `luna_model` model; most algorithms turn it into a QUBO
(`lp_to_qubo` in `bench/building_blocks.py`), but you are free to encode the model differently.

| size class | QUBO variables | instances | typical run time per algorithm and instance |
|---|---|---|---|
| small | up to 8 | 10 | seconds |
| mid | up to 16 | 27 | seconds to minutes |
| large | up to 20 | 7 | minutes |
| very-large | up to 24 | 3 | up to the time limit |

`--size` includes all smaller classes. Work on the small instances first, they give feedback in minutes.

The final ranking uses new, hidden instances: don't tune your algorithm to, or hard-code anything from, the
instances in `bench/data/instances.yaml`.

## Rules and Configuration

Fixed for everyone in `bench/rules.py` and `bench/config.py`, enforced by `bench/execution.py`:

- Circuits run on a noiseless statevector simulator with all-to-all coupling (`backend="ideal"`) and must be
  executed with `bench.execution.ExecutionService.execute`, which records them for the metrics (see the
  examples in `hackathon/teams/example/`).
- At most **25 qubits** per circuit, ancillas included.
- At most **120 s** per run of a local algorithm: everything `run_with_logging` does counts, including the
  final sampling and `backtransform` (a run that breaks the limit returns no samples). Stop in time: `minimize`
  in the example team's `utils.py` does it with `time_left=execution_service.remaining_time`; a loop of your own
  checks `execution_service.remaining_time()` before every step that executes circuits.
- **The final circuit**: end the run with exactly one call of `execution_service.execute_last_and_return_solution`
  (or of `execution_service.execute_final`, which returns the counts, if you use an encoding of your own and map
  them to a solution yourself). It executes the final circuit with exactly **1024 shots**, whose samples are the
  solution; no circuit may run after it. A run that breaks this ends unsolved.
- At most **2 algorithms per team** are ranked: the first two entries of `ALGORITHMS` by name. Before handing in,
  benchmark as many as you like.
- Don't parallelize your own code: no multi-threading, multi-processing or similar.
- **Solutions come from quantum circuits.** The core of the algorithm must be quantum, and its classical
  processing must scale with the problem size. Classical code may build circuits from the problem's
  coefficients and constraints, optimize circuit parameters, and evaluate samples that a circuit returned,
  e.g. as the optimizer's cost. It may not compute or improve candidate solutions itself: no brute force,
  greedy or local search, simulated annealing or exact solver on the model or its QUBO, and no circuit that only
  writes down a classically computed answer.
- **Warm starts only from measured samples.** A circuit may start from a sample that an earlier circuit of
  the same run returned (iterative warm start, recursive QAOA). A warm start from a classical or heuristic
  precomputation is not allowed. In recursive QAOA, the remaining small problem is sampled by a circuit as
  well, not solved classically.
- **The final circuit must not merely prepare a measured sample**, e.g. X gates on the 1-bits of the best sample
  of an earlier circuit.
- The mentors read the code of the top entries. An entry that breaks these rules is not ranked.
- For the metrics, every circuit is additionally transpiled onto the real device FakeHanoiV2 (IBM, 27 qubits:
  coupling map, native gates, gate durations). That circuit is not simulated and does not count towards the
  time limit.

## Metrics

| metric | meaning |
|---|---|
| approximation ratio | per sample `1 - abs(f - f_opt) / abs(f_opt)`, bounded to [0, 1]; over all samples with infeasible ones counted as 0, and over the feasible samples only |
| feasibility ratio | share of samples that satisfy all constraints |
| P(opt), time to solution | share of optimal samples; wall-clock time until the optimum is found with 99 % |
| TTS (layers) | the main metric, as FlexQAOA reports it: `L x ceil(log(1 - 0.99) / log(1 - P(opt)))`, with `L` the layers of the final circuit on an all-to-all device; the optimization loop is not counted |
| QPU time, QPU time to solution | estimated time on FakeHanoiV2: shots x (circuit duration + 250 us) for every circuit of the run; time to solution as above |
| shots to solution | the same counted in shots |
| circuit statistics | qubits, depth and two-qubit gates on FakeHanoiV2 and on the ideal backend, number of circuits, shots |

The optimum `f_opt` of every instance is computed with SCIP when the benchmark runs.

## Your team's folder

Every team works in its own folder, `hackathon/teams/<team>/`. After the event, the mentors merge all teams
into one project and benchmark them together.

1. Pick a team name that is a Python identifier, e.g. `qubit_crew`; it is your `<team>` below. Create your
   team's branch: `git switch -c team/<team>`. Commit only there; `main` stays the common starting point.
   Then copy `hackathon/teams/example/` to `hackathon/teams/<team>/`.
2. In your copy, delete the example files you don't use, rename the classes and set every `algorithm_id` to
   `"<team>.<ClassName>"`; the benchmark rejects any other prefix. List your algorithms in `ALGORITHMS` of your
   `__init__.py`: an entry `"<name>": YourClass(...)` appears as `<team>-<name>` in results and on the command
   line.
3. Put your idea into `run_with_logging`: `my_algorithm.py` is a working template (random sampling),
   `qaoa.py` a textbook QAOA, `custom_vqe.py` a VQE. All three show how to run circuits with `ExecutionService`,
   and `qaoa.py` marks ideas for first improvements as hints.
4. Try it on one instance: `uv run python examples/solve_one_problem.py traffic_c3 <team>-<name>`.
5. Benchmark it: `uv run python run_benchmark.py --only <team>` runs all your algorithms; the others keep their
   stored results.
6. Need a package? `uv add --group <team> <package>` puts it into your own dependency group; `uv sync`
   installs it.
7. Check: `uv run pytest` also runs each of your algorithms on a 3-variable MaxCut and expects the optimum
   among the samples.
8. Hand in: commit your folder, `pyproject.toml` and `uv.lock` on `team/<team>`. Then choose one of three ways:
   - This repository: `git push -u origin team/<team>`. Ask the mentors for write access first. Note that every
     other team can read your branch here.
   - A private GitHub repository of your team: create an empty private repository (not a fork: a fork of a public
     repository is public), add it with `git remote add team <url>`, push with `git push -u team team/<team>`,
     and invite the mentors' GitHub account as collaborator (the mentors tell you its name).
   - A bundle file: `git bundle create <team>.bundle main..team/<team>`; give the file to the mentors.

Change nothing outside your folder, except your group in `pyproject.toml` and `uv.lock`, which uv writes. The
mentors check every submission with `bench/check_submission.py`, which rejects other changes.

`bench/building_blocks.py` has the fixed building blocks (QUBO and Ising form, cost functions such as CVaR);
copy one into your folder to change it, or add another transformation or encoding (see "Good to know"), and
update the imports. `utils.py` in your folder has the ones you will likely adapt (initial state, measurements,
optimizer with time limit). Ideas worth trying: QAOA, VQE or digitized annealing as a start, then better
initial parameters or schedules, constraint-preserving mixers, warm starts from measured samples, recursion,
... (see also the hints for first improvements, `Hints/PushQuantum_2026_hints.pdf`).

## Project structure

```
run_benchmark.py          Luna baselines plus every team's algorithms
examples/                 solve_one_problem.py: one instance, one algorithm, the benchmark metrics
hackathon/teams/          one folder per team
  example/                copy it to start your team: the template (my_algorithm.py), a textbook QAOA
                          (qaoa.py), a VQE (custom_vqe.py) and utils.py
bench/                    benchmark infrastructure (rules, metrics, figures); please do not change
  rules.py                qubit and time limit, final shots, algorithms per team, size classes
  config.py               seeds, metrics, figures
  registry.py             finds the teams' algorithms
  check_submission.py     the mentors' check of a submission
  data/instances.yaml     the instances
Hints/                    hints for first improvements (PDF)
Papers/                   papers referenced in the hints
```

## Ranking

The ranking criteria involve the following:

| Criterion | Metric |
|---|---|
| Algorithm performance | Primary: instances solved. Secondary: TTS (layers) |
| Presentation quality | Quality of the solution pitch |
| Algorithm idea | Novelty and creativity of the idea |

The benchmark for the ranking runs on CPU only, on a MacBook Pro (Apple M2, 16 GB RAM, 8 CPU cores: 4
performance and 4 efficiency).

## Good to know

- luna_bench stores all results in `benchmark_data/`; a new run only computes what is missing or changed,
  plus failed or interrupted runs (`--rerun NAME` repeats an unchanged algorithm, `--no-retry` skips failed runs,
  `--clean` deletes all stored results first and computes everything again).
- If you abort a benchmark (Ctrl+C), delete `benchmark_data/database/luna_bench-jobs.db` before the next run,
  otherwise luna_bench first executes the queued jobs of the aborted run.
- A stored benchmark can't be loaded once one of its algorithms is gone: after you rename an `algorithm_id` or
  delete an algorithm, run under a new `--name`, or start fresh: move `benchmark_data/` aside or run with
  `--clean`. Benchmarks stored before the team folders hold `hackathon.MyAlgorithm` and need the same.
- On Windows, set `PYTHONUTF8=1` if the console output fails with a `UnicodeEncodeError`.
- `--seed N` fixes the seeds of the transpiler, the simulator and the metrics transpilation; for
  `solve_one_problem.py`, set the environment variable `BENCH_SEED=N` instead. The final ranking uses a seed
  you don't know. Random numbers of your own algorithm, e.g. NumPy's, are yours to seed. A new seed doesn't
  repeat stored runs: add `--rerun <team>` or `--clean`.
- You are not restricted to QUBO: you can turn the model into a circuit in other ways, e.g. from a binary
  minimization (similar to the QUBO transformation in `building_blocks.py`). Put it into your `utils.py` and end
  the run with `execute_final` (see the rules). You then do yourself what `execute_last_and_return_solution` does
  for a QUBO: build the `Solution` from all 1024 counts (`backtransform` in `bench/utils.py` shows how) and report
  the optimizer's iterations with `execution_service.record_optimizer_iterations`. Make sure your encoding doesn't
  solve the problem classically.
