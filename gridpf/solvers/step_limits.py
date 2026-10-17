"""Newton step-size limiting and the post-step voltage/angle corridor.

Two independent safeguards on top of the residual-norm backtracking in
:func:`gridpf.solvers.newton_raphson.newton_raphson`. Both act on the whole
Newton increment through a single scalar factor, so the Newton direction is
preserved.

1. **Step scaling** (before the step): the increment ``Δx`` is scaled by
   ``α = min(1, limit_k / |Δ_k|)`` over all constraints —

   * relative modulus step weighted by the voltage level:
     ``|Δ|V|| / |V| / |V|`` (``|V|`` in p.u. of nominal, so the limit gets
     tighter at low voltage);
   * per-bus angle step ``|Δδ_i|``;
   * per-branch angle-difference step ``|Δδ_i − Δδ_j|``.

   Backtracking, when enabled, then starts from ``α`` instead of ``1``.

2. **Corridor correction** (after the step, from the second iteration on):
   if a bus modulus left ``[v_lo, v_hi]`` or a branch angle difference
   exceeded ``max_branch_angle`` *while moving outward* in this step, the
   whole applied step is rolled back by one factor so that the worst
   violator lands exactly on the boundary; the other buses move back
   proportionally. One correction per iteration, no repetition.

Branches are taken from the off-diagonal pattern of ``Ybus`` (parallel
branches collapse into one bus pair, which is what the angle limits need).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class StepLimits:
    """Step-size limits of the Newton iteration; ``None`` disables a limit.

    Attributes:
        max_dvm: limit on ``|Δ|V|| / |V|²`` (relative modulus step divided by
            ``|V|`` in p.u.), dimensionless.
        max_dva: limit on the per-bus angle step ``|Δδ_i|``, rad.
        max_branch_dva: limit on the per-branch angle-difference step
            ``|Δδ_i − Δδ_j|``, rad.
        v_corridor: ``(v_lo, v_hi)`` — post-step corridor on ``|V|``, p.u.
        max_branch_angle: post-step limit on ``|δ_i − δ_j|`` over branches, rad.
    """

    max_dvm: float | None = None
    max_dva: float | None = None
    max_branch_dva: float | None = None
    v_corridor: tuple[float, float] | None = None
    max_branch_angle: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_dvm", "max_dva", "max_branch_dva", "max_branch_angle"):
            value = getattr(self, name)
            if value is not None and not value > 0.0:
                raise ValueError(f"step limit {name} must be > 0, got {value}")
        if self.v_corridor is not None:
            v_lo, v_hi = self.v_corridor
            if not 0.0 <= v_lo < v_hi:
                raise ValueError(
                    f"step_v_corridor must satisfy 0 <= v_lo < v_hi, got {self.v_corridor}"
                )

    @property
    def scaling_active(self) -> bool:
        """Whether any pre-step scaling limit is set."""
        return (
            self.max_dvm is not None or self.max_dva is not None or self.max_branch_dva is not None
        )

    @property
    def corridor_active(self) -> bool:
        """Whether any post-step corridor limit is set."""
        return self.v_corridor is not None or self.max_branch_angle is not None

    @property
    def needs_branches(self) -> bool:
        """Whether the bus-pair list of the network is required."""
        return self.max_branch_dva is not None or self.max_branch_angle is not None


def branch_pairs(Ybus: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(i, j)`` bus pairs (``i < j``) coupled by at least one branch."""
    coo = Ybus.tocoo()
    upper = coo.row < coo.col
    return coo.row[upper].astype(np.int64), coo.col[upper].astype(np.int64)


def expand_increment(
    dx: np.ndarray, n_bus: int, pv: np.ndarray, pq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spread the Newton increment ``[Δδ(pv); Δδ(pq); Δ|V|(pq)]`` over all buses."""
    n_pv = pv.size
    n_pq = pq.size
    dva = np.zeros(n_bus, dtype=np.float64)
    dvm = np.zeros(n_bus, dtype=np.float64)
    dva[pv] = dx[:n_pv]
    dva[pq] = dx[n_pv : n_pv + n_pq]
    dvm[pq] = dx[n_pv + n_pq :]
    return dva, dvm


def step_scale(
    Vm: np.ndarray,
    dva: np.ndarray,
    dvm: np.ndarray,
    limits: StepLimits,
    pairs: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    """Largest ``α ≤ 1`` such that ``α·Δx`` satisfies every scaling limit."""
    alpha = 1.0
    if limits.max_dvm is not None and dvm.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(dvm) / (Vm * Vm)
        worst = float(np.nanmax(rel))
        if worst > limits.max_dvm:
            alpha = min(alpha, limits.max_dvm / worst)
    if limits.max_dva is not None and dva.size:
        worst = float(np.max(np.abs(dva)))
        if worst > limits.max_dva:
            alpha = min(alpha, limits.max_dva / worst)
    if limits.max_branch_dva is not None and pairs is not None and pairs[0].size:
        worst = float(np.max(np.abs(dva[pairs[0]] - dva[pairs[1]])))
        if worst > limits.max_branch_dva:
            alpha = min(alpha, limits.max_branch_dva / worst)
    return alpha


def corridor_correction(
    Vm_old: np.ndarray,
    Va_old: np.ndarray,
    Vm_new: np.ndarray,
    Va_new: np.ndarray,
    limits: StepLimits,
    pairs: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    """Fraction ``a ∈ [−1, 0]`` of the applied step to roll back.

    Only outward motion counts as a violation: a bus already outside the
    corridor that moves back towards it is left alone. ``0.0`` means no
    correction.
    """
    a = 0.0
    if limits.v_corridor is not None:
        v_lo, v_hi = limits.v_corridor
        d = Vm_new - Vm_old
        up = (Vm_new > v_hi) & (d > 0.0)
        if up.any():
            a = min(a, float(np.min((v_hi - Vm_new[up]) / d[up])))
        down = (Vm_new < v_lo) & (d < 0.0)
        if down.any():
            a = min(a, float(np.min((v_lo - Vm_new[down]) / d[down])))
    if limits.max_branch_angle is not None and pairs is not None and pairs[0].size:
        i, j = pairs
        th_new = np.abs(Va_new[i] - Va_new[j])
        th_old = np.abs(Va_old[i] - Va_old[j])
        dth = np.abs((Va_new[i] - Va_new[j]) - (Va_old[i] - Va_old[j]))
        out = (th_new > limits.max_branch_angle) & (th_new > th_old) & (dth > 0.0)
        if out.any():
            a = min(a, float(np.min((limits.max_branch_angle - th_new[out]) / dth[out])))
    return max(a, -1.0)
