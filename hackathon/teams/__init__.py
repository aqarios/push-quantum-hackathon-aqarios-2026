"""One package per team, ``hackathon/teams/<team>/``, with the team's algorithms in ``ALGORITHMS`` of its
``__init__.py``.

``bench.registry.discover`` imports every package here, so the benchmark, the example and the tests need no list of
teams. A new team starts as a copy of ``example/``; the README has the steps.
"""
