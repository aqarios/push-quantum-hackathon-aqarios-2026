"""Check a team's submission on the mentors' notebook before it is merged (there is no CI).

    uv run python -m bench.check_submission alpha submissions/alpha                 # the team's branch against main
    uv run python -m bench.check_submission alpha submissions/alpha --base compare  # against another branch

Three checks, one line each:

1. Files: ``git diff <base>...<ref>`` changes only ``hackathon/teams/<team>/``, ``pyproject.toml`` and ``uv.lock``.
2. ``pyproject.toml``: everything except the team's own dependency group is the same as on ``<base>``.
3. Algorithms: discovery finds the team's algorithms, and each one returns samples on ``--instance``. This check runs
   the code that is checked out, so merge the submission (or check out ``<ref>``) and run ``uv sync`` first.

Exits with 1 if a check fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib

SHARED_FILES = ("pyproject.toml", "uv.lock")


def git(*args: str) -> str:
    """Output of a git command in the current directory."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def check_files(team: str, base: str, ref: str) -> tuple[bool, str]:
    """Only the team folder and the two shared project files may change."""
    files = git("diff", "--name-only", f"{base}...{ref}").split()
    outside = [file for file in files if not file.startswith(f"hackathon/teams/{team}/") and file not in SHARED_FILES]
    if outside:
        return False, f"files: changed outside hackathon/teams/{team}/: {', '.join(outside)}"
    return True, f"files: {len(files)} changed, all in hackathon/teams/{team}/ or {' / '.join(SHARED_FILES)}"


def check_pyproject(team: str, base: str, ref: str) -> tuple[bool, str]:
    """Apart from the dependency group named after the team, pyproject.toml is unchanged."""

    def without_team_group(rev: str) -> tuple[dict, list[str]]:
        project = tomllib.loads(git("show", f"{rev}:pyproject.toml"))
        return project, project.get("dependency-groups", {}).pop(team, [])

    before, _ = without_team_group(base)
    after, packages = without_team_group(ref)
    changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    if "dependency-groups" in changed:
        groups_before, groups_after = before.get("dependency-groups", {}), after.get("dependency-groups", {})
        changed.remove("dependency-groups")
        changed += [f"dependency-groups.{group}" for group in sorted(set(groups_before) | set(groups_after))
                    if groups_before.get(group) != groups_after.get(group)]
    if changed:
        return False, f"pyproject.toml: changed outside the group {team!r}: {', '.join(changed)}"
    return True, f"pyproject.toml: only the group {team!r} may differ; it has {packages or 'no packages'}"


def check_algorithms(team: str, instance: str) -> list[tuple[bool, str]]:
    """Every algorithm of the team, as discovery finds it in the checked-out tree, returns samples on ``instance``."""
    from bench import rules  # noqa: PLC0415
    from bench.registry import TeamError, discover, dropped_algorithms  # noqa: PLC0415 - imports luna_bench
    from bench.utils import load_instances  # noqa: PLC0415

    try:
        algorithms = {name: algorithm for name, algorithm in discover().items() if name.startswith(f"{team}-")}
    except TeamError as error:
        return [(False, f"discovery: {error}")]
    if not algorithms:
        return [(False, f"discovery: no algorithms of team {team!r}; merge the submission first, then uv sync")]
    results = []
    dropped = dropped_algorithms(team)
    if dropped:  # not a failure: the ranking uses the first ones by name
        results.append((True, f"algorithms: more than {rules.MAX_ALGORITHMS_PER_TEAM}; ranked: "
                              f"{', '.join(sorted(algorithms))}; dropped: {', '.join(dropped)}"))
    model = {problem["name"]: model for problem, model in load_instances()}[instance]
    for name, algorithm in algorithms.items():
        start = time.perf_counter()
        try:
            solution = algorithm.run(model)
        except Exception as error:  # noqa: BLE001 - a team's code may raise anything; report it and go on
            results.append((False, f"{name}: fails on {instance}: {error}"))
            continue
        seconds = time.perf_counter() - start
        ok = len(solution.samples) > 0
        results.append((ok, f"{name}: {len(solution.samples)} distinct samples on {instance} in {seconds:.1f} s"
                        + ("" if ok else " (an empty solution: the run broke a limit or the final-shot rule)")))
    return results


def main() -> None:
    """Run the three checks and exit with 1 if one fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("team", help="team name: folder hackathon/teams/<team>/ and dependency group <team>")
    parser.add_argument("ref", help="the submitted git branch or commit")
    parser.add_argument("--base", default="main", help="branch the submission is checked against (default: main)")
    parser.add_argument("--instance", default="traffic_c2",
                        help="small instance for the algorithm check (default: traffic_c2, 6 qubits, one-hot)")
    parser.add_argument("--time-limit", type=float, default=120,
                        help="seconds per run in the algorithm check (default: 120)")
    args = parser.parse_args()
    os.environ["BENCH_TIME_LIMIT"] = str(args.time_limit)

    print(f"Submission of team {args.team!r}: {args.ref} against {args.base}")
    results = [check_files(args.team, args.base, args.ref), check_pyproject(args.team, args.base, args.ref),
               *check_algorithms(args.team, args.instance)]
    for ok, message in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {message}")
    sys.exit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()
