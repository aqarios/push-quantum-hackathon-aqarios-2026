"""Building blocks for the algorithms, fixed for all teams.

* QUBO: :func:`lp_to_qubo` turns a constrained model into a QUBO matrix, :func:`counts_to_bits` turns Qiskit
  counts into a 0/1 matrix.
* Cost functions on the counts: :func:`expectation` is the mean QUBO value, :func:`cvar` the mean of the best
  shots, :func:`correlations` the mean ``<Z_i>`` and ``<Z_i Z_j>``.
* Ising form: :func:`qubo_to_ising`, :func:`scaled_ising`, :func:`ising_operator`, and :func:`pair_rounds` to
  apply commuting two-qubit gates in few layers.

Do not change these functions: the benchmark maps every final sample back to the model with
:func:`counts_to_bits` (``bench.utils.backtransform``), and all teams share them. To change one, copy it into your
team folder; your own cost functions go there too. The building blocks you will likely adapt (initial state,
measurements, optimizer) are in ``utils.py`` of your team folder, a copy of ``hackathon/teams/example/utils.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rustworkx
from luna_model.transformation import PassManager, pipelines
from luna_model.translator import QuboTranslator
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from luna_model import Model
    from luna_model.transformation import TransformationOutput


# --------------------------------------------------------------------------------------
# QUBO
# --------------------------------------------------------------------------------------


def lp_to_qubo(model: Model, penalty_scaling: float = 10.0) -> tuple[np.ndarray, TransformationOutput]:
    """QUBO matrix ``Q`` of a constrained model and the transformation that produced it.

    The QUBO is a minimization of ``x^T Q x`` (``Q`` symmetric, constant offset dropped); variable
    ``i`` is meant to live on qubit ``i``. Constraints become quadratic penalties, scaled by the
    largest objective coefficient times ``penalty_scaling``; inequalities get binary slack variables.
    Keep the transformation for :func:`backtransform`.
    """
    transformation = PassManager([pipelines.ToUnconstrainedBinaryPipeline(penalty_scaling=penalty_scaling)]).run(model)
    return QuboTranslator.from_lm(transformation.model).matrix, transformation


def counts_to_bits(counts: Mapping[str, int], num_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Qiskit counts as a 0/1 matrix (one row per bitstring, column ``i`` = classical bit ``i``) and frequencies."""
    keys = list(counts)
    bits = np.array([[int(key[-1 - i]) for i in range(num_bits)] for key in keys], dtype=np.int8)
    return bits.reshape(len(keys), num_bits), np.array([counts[key] for key in keys])


# --------------------------------------------------------------------------------------
# Cost functions on the counts
# --------------------------------------------------------------------------------------


def expectation(qubo: np.ndarray, counts: Mapping[str, int]) -> float:
    """Mean QUBO value ``x^T Q x`` over all shots."""
    bits, freqs = counts_to_bits(counts, len(qubo))
    values = np.einsum("ki,ij,kj->k", bits, qubo, bits)
    return float(values @ freqs / freqs.sum())


def cvar(qubo: np.ndarray, counts: Mapping[str, int], alpha: float = 1.0) -> float:
    """Mean QUBO value of the best ``alpha`` share of the shots (conditional value at risk, Barkoutsos et al.,
    Quantum 4, 256 (2020)); ``alpha = 1`` is :func:`expectation`."""
    bits, freqs = counts_to_bits(counts, len(qubo))
    values = np.einsum("ki,ij,kj->k", bits, qubo, bits)
    order = np.argsort(values)
    values, freqs = values[order], freqs[order]
    before = np.concatenate([[0], np.cumsum(freqs)[:-1]])  # shots with a lower value
    taken = np.clip(alpha * freqs.sum() - before, 0, freqs)  # shots of each value within the best alpha share
    return float(values @ taken / taken.sum())


def correlations(counts: Mapping[str, int], num_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``<Z_i>`` (vector) and ``<Z_i Z_j>`` (matrix) over all shots, with ``Z = 1 - 2 x`` per bit."""
    bits, freqs = counts_to_bits(counts, num_bits)
    z = 1 - 2 * bits.astype(float)
    weights = freqs / freqs.sum()
    return weights @ z, z.T @ (weights[:, None] * z)


# --------------------------------------------------------------------------------------
# Ising form: x = (1 - z) / 2 turns the QUBO into H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
# --------------------------------------------------------------------------------------


def qubo_to_ising(qubo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fields ``h`` and upper-triangular couplings ``J`` of ``x^T Q x`` with ``x = (1 - z) / 2`` (constant dropped)."""
    couplings = np.triu(qubo + qubo.T, k=1)  # coefficient of x_i x_j, i < j
    h = -np.diag(qubo) / 2 - (couplings.sum(axis=0) + couplings.sum(axis=1)) / 4
    return h, couplings / 4


def scaled_ising(qubo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`qubo_to_ising` divided by its largest absolute coefficient, so angles and times mean the same
    on every instance."""
    h, j = qubo_to_ising(qubo)
    scale = np.abs(np.concatenate([h, j.ravel()])).max() or 1.0
    return h / scale, j / scale


def ising_operator(h: np.ndarray, j: np.ndarray) -> SparsePauliOp:
    """``H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j`` as a Qiskit operator (qubit ``i`` = variable ``i``)."""
    terms = [("Z", [int(i)], h[i]) for i in np.flatnonzero(h)]
    terms += [("ZZ", [int(a), int(b)], j[a, b]) for a, b in zip(*np.nonzero(j), strict=True)]
    return SparsePauliOp.from_sparse_list(terms or [("", [], 0.0)], num_qubits=len(h))


def pair_rounds(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """``pairs`` split into rounds of disjoint pairs (edge coloring, Misra-Gries: at most max degree + 1)."""
    if not pairs:
        return []
    graph = rustworkx.PyGraph()
    graph.add_nodes_from(range(1 + max(max(pair) for pair in pairs)))
    graph.add_edges_from_no_data(pairs)
    rounds: dict[int, list[tuple[int, int]]] = {}
    for index, color in rustworkx.graph_misra_gries_edge_color(graph).items():
        rounds.setdefault(color, []).append(pairs[index])
    return [rounds[color] for color in sorted(rounds)]
