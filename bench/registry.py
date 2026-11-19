"""The teams' algorithms, found in ``hackathon/teams/``: one package per team, each with an ``ALGORITHMS`` dict.

:func:`discover` imports every team package and names each algorithm ``<team>-<name>``. ``run_benchmark.py``,
``examples/solve_one_problem.py``, the tests and ``bench.check_submission`` use it, so a new team edits no shared
file.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from bench import rules
from bench.execution import HackathonParticipantAlgorithm

logger = logging.getLogger(__name__)

TEAMS_PACKAGE = "hackathon.teams"
# Teams exempt from rules.MAX_ALGORITHMS_PER_TEAM: the mentors' reference algorithms.
UNLIMITED_TEAMS = ("example",)


class TeamError(ValueError):
    """A team package breaks a rule of the team layout; the message names the team and the rule."""


def discover(package: str = TEAMS_PACKAGE) -> dict[str, HackathonParticipantAlgorithm]:
    """The algorithms of every team package in ``package``, as ``{"<team>-<name>": algorithm}``, sorted by team.

    A team with more than ``rules.MAX_ALGORITHMS_PER_TEAM`` algorithms keeps only the first ones by name (sorted);
    the others are dropped with a warning. The teams in ``UNLIMITED_TEAMS`` are exempt.

    Raises :class:`TeamError` if a folder in ``package`` is not an importable package named by a Python identifier,
    if a team has no ``ALGORITHMS`` dict of :class:`~bench.execution.HackathonParticipantAlgorithm` instances, or if a
    class's ``algorithm_id`` does not start with ``"<team>."``.
    """
    root = Path(importlib.import_module(package).__file__).parent
    algorithms: dict[str, HackathonParticipantAlgorithm] = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "__pycache__"):
        team = folder.name
        if not team.isidentifier() or not (folder / "__init__.py").is_file():
            msg = f"{folder}: a team folder needs a Python identifier as its name and an __init__.py"
            raise TeamError(msg)
        team_algorithms = getattr(importlib.import_module(f"{package}.{team}"), "ALGORITHMS", None)
        if not isinstance(team_algorithms, dict):
            msg = f"team {team!r}: {package}.{team} must define ALGORITHMS = {{name: algorithm, ...}}"
            raise TeamError(msg)
        kept = set(sorted(team_algorithms)[:rules.MAX_ALGORITHMS_PER_TEAM]) if team not in UNLIMITED_TEAMS \
            else set(team_algorithms)
        dropped = [name for name in sorted(team_algorithms) if name not in kept]
        if dropped:
            logger.warning("team %r: more than %d algorithms; using %s, dropping %s", team,
                           rules.MAX_ALGORITHMS_PER_TEAM, sorted(kept), dropped)
        for name, algorithm in team_algorithms.items():
            checked = _checked(team, name, algorithm)
            if name in kept:
                algorithms[f"{team}-{name}"] = checked
    return algorithms


def dropped_algorithms(team: str, package: str = TEAMS_PACKAGE) -> list[str]:
    """Names in the team's ``ALGORITHMS`` that :func:`discover` drops (more than ``MAX_ALGORITHMS_PER_TEAM``)."""
    if team in UNLIMITED_TEAMS:
        return []
    names = sorted(getattr(importlib.import_module(f"{package}.{team}"), "ALGORITHMS", {}))
    return names[rules.MAX_ALGORITHMS_PER_TEAM:]


def _checked(team: str, name: str, algorithm: object) -> HackathonParticipantAlgorithm:
    if not isinstance(algorithm, HackathonParticipantAlgorithm):
        msg = f"team {team!r}: ALGORITHMS[{name!r}] is not a HackathonParticipantAlgorithm"
        raise TeamError(msg)
    # The team prefix keeps algorithm ids unique across teams. luna_bench registers only the first class of an id
    # and silently ignores the others, so a copied template that kept "example.MyAlgorithm" would run, and be
    # reported, as the example team's class.
    algorithm_id = getattr(type(algorithm), "registered_id", None)  # set by @algorithm
    if not (algorithm_id or "").startswith(f"{team}."):
        msg = (f"team {team!r}: {type(algorithm).__name__} has algorithm_id {algorithm_id!r}; it must start with "
               f"{team + '.'!r}, e.g. @algorithm(algorithm_id=\"{team}.{type(algorithm).__name__}\")")
        raise TeamError(msg)
    return algorithm
