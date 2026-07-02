"""Shared building blocks for the iterative PF solvers.

Holds the pieces that Newton-Raphson and Gauss-Seidel used to duplicate:
the result dataclass, the complex power mismatch, the active/reactive
residual vector, its infinity norm, and the voltage-dependent-load gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


@dataclass
class SolverResult:
    """Result of an iterative PF solver run."""

    V: np.ndarray
    converged: bool
    iterations: int
    mismatch_max: float


def mismatch(Ybus: csr_matrix, V: np.ndarray, Sbus: np.ndarray) -> np.ndarray:
    """Полный комплексный небаланс ``V · conj(Ybus · V) − Sbus``."""
    result: np.ndarray = V * np.conj(Ybus @ V) - Sbus
    return result


def residual_vector(mis: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> np.ndarray:
    """Активная часть небаланса ``F = [Re(mis[pvpq]); Im(mis[pq])]``."""
    pvpq = np.concatenate([pv, pq])
    return np.concatenate([mis[pvpq].real, mis[pq].imag])


def residual_norm(f: np.ndarray) -> float:
    """Infinity norm of the residual vector (0.0 when empty)."""
    return float(np.linalg.norm(f, np.inf)) if f.size else 0.0


def resolve_use_load(net: PFInput | None, voltage_dependent_load: bool) -> bool:
    """Return whether voltage-dependent load handling is active for this run."""
    return voltage_dependent_load and net is not None and net.has_voltage_dependent_load
