"""Ordered Gauss-Seidel warm start (``PFOptions.gs_mode="ordered"``).

A fixed number of Gauss-Seidel passes that replaces the first Newton
iteration, robust on networks with very small series impedances (breakers):

* **Update order** — built once by Dijkstra from all slack buses at once over
  the series impedance ``|z| = |r + jx|`` of the branches, referred through
  the transformer ratios along the path (factor ``s``, ``s = 1`` at a slack):

  - line (``|t| = 1``, no phase shift): weight ``s²·|z|``, ``s`` unchanged;
  - transformer, step from its "from" side: ``s²·|z|``, ``s ← s/|t|``;
  - transformer, step from its "to" side: ``|t|²·|z|``, ``s ← s·|t|``.

  Buses are updated by increasing distance, PQ and PV mixed; buses joined by
  a near-zero impedance therefore come right after each other. Buses not
  reachable from a slack are not updated.
* **Load as admittance** — on a non-slack bus whose load at the current
  ``|V|`` has ``P_load > 0`` and ``Q_load > 0``, the load enters the diagonal
  as ``Y_kk + (P_load − jQ_load)/|V|²`` (``|V|`` from the previous pass) and
  only the generation stays an injection. The base ``Ybus`` is untouched.
* **Series capacitances shorted** — a branch whose series admittance
  ``g + jb = 1/(r + jx)`` has ``b > 0`` (``x < 0``, e.g. a star branch of a
  three-winding transformer) gets ``b := −1/x_eq`` with the small inductive
  reactance ``x_eq = 55·min_x·U_from`` Ω (``U_from`` — nominal voltage of its
  "from" bus, kV; ``g`` kept) in a Gauss-Seidel-only copy of ``Ybus``; the order is built
  from the original parameters and Newton keeps the original ``Ybus``.
* PV buses: ``Q`` from the balance at the current voltages, the voltage
  update, then ``|V| = V_set`` with the angle kept. ``ω = 1``; a non-finite
  update resets the bus to ``1∠0``.

There is no convergence criterion: exactly ``passes`` passes are made.
"""

from __future__ import annotations

import heapq
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridpf.algebra.sbus import compute_sbus
from gridpf.algebra.ybus import build_ybus
from gridpf.solvers._common import (
    SolverResult,
    mismatch,
    residual_norm,
    residual_vector,
    resolve_use_load,
)


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


# Ohm per (kV · unit of min_x) of the substitute reactance x_eq = 55·min_x·U.
_GS_SHORT_X_OHM_PER_KV = 55.0


def _gs_ybus(net: PFInput, Ybus: csr_matrix, min_x: float) -> csr_matrix:
    """``Ybus`` with series-capacitive branches shorted (Gauss-Seidel only).

    In p.u. ``x_eq = 55·min_x·U·S_base/U² = 55·min_x·S_base/U`` (``U`` in kV).
    """
    r = np.asarray(net.branch_r, dtype=np.float64)
    x = np.asarray(net.branch_x, dtype=np.float64)
    y = 1.0 / (r + 1j * x)
    cap = y.imag > 0.0
    if not cap.any():
        return Ybus
    u_from = np.asarray(net.bus_vn_kv, dtype=np.float64)[np.asarray(net.from_idx)]
    y_new = y.copy()
    x_eq = _GS_SHORT_X_OHM_PER_KV * min_x * net.base_mva / u_from[cap]
    y_new[cap] = y[cap].real - 1j / x_eq
    z_new = 1.0 / y_new
    shorted = replace(net, branch_r=z_new.real, branch_x=z_new.imag)
    return build_ybus(shorted)[0]


def traversal_order(net: PFInput, ref: np.ndarray) -> np.ndarray:
    """Non-slack buses reachable from a slack, by increasing referred ``|z|``."""
    n = net.n_bus
    z = np.abs(np.asarray(net.branch_r) + 1j * np.asarray(net.branch_x))
    t = np.abs(np.asarray(net.tap_ratio, dtype=np.float64))
    xfmr = (t != 1.0) | (np.asarray(net.phase_shift) != 0.0)
    adj: list[list[tuple[int, int, bool]]] = [[] for _ in range(n)]
    for b, (f, to) in enumerate(zip(net.from_idx.tolist(), net.to_idx.tolist(), strict=True)):
        adj[f].append((to, b, True))
        adj[to].append((f, b, False))

    dist = np.full(n, np.inf)
    scale = np.ones(n)
    done = np.zeros(n, dtype=bool)
    heap: list[tuple[float, int]] = []
    for k in ref.tolist():
        dist[k] = 0.0
        heapq.heappush(heap, (0.0, k))
    order: list[int] = []
    is_ref = np.zeros(n, dtype=bool)
    is_ref[ref] = True
    while heap:
        d, k = heapq.heappop(heap)
        if done[k]:
            continue
        done[k] = True
        if not is_ref[k]:
            order.append(k)
        s = scale[k]
        for nb, b, from_side in adj[k]:
            if done[nb]:
                continue
            if not xfmr[b]:
                w, s_nb = s * s * z[b], s
            elif from_side:
                w, s_nb = s * s * z[b], s / t[b]
            else:
                w, s_nb = t[b] * t[b] * z[b], s * t[b]
            if d + w < dist[nb]:
                dist[nb] = d + w
                scale[nb] = s_nb
                heapq.heappush(heap, (d + w, nb))
    return np.array(order, dtype=np.int64)


def gauss_seidel_ordered(
    Ybus: csr_matrix,
    Sbus: np.ndarray,
    V0: np.ndarray,
    ref: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    *,
    net: PFInput,
    passes: int,
    tol: float,
    min_x: float = 4e-6,
    voltage_dependent_load: bool = False,
) -> SolverResult:
    """Run ``passes`` ordered Gauss-Seidel passes from ``V0``.

    Args:
        Ybus: ``(n, n)`` CSR.
        Sbus: ``(n,)`` complex — injections under the current classification
            (used for buses whose load is not turned into an admittance).
        V0: starting voltage.
        ref, pv, pq: bus classification.
        net: the network (branches for the order, load/generation split).
        passes: number of passes (no convergence criterion).
        tol: only sets the ``converged`` flag of the result.
        min_x: scale of the reactance shorting series-capacitive branches,
            ``x_eq = 55·min_x·U_from[kV]`` Ω.
        voltage_dependent_load: whether the load model depends on ``|V|``.
    """
    if passes < 0:
        raise ValueError(f"passes must be >= 0, got {passes}")
    use_load = resolve_use_load(net, voltage_dependent_load)
    V = V0.astype(np.complex128, copy=True)
    order = traversal_order(net, ref)
    is_pv = np.zeros(V.size, dtype=bool)
    is_pv[pv] = True
    v_set = np.abs(V0)

    split = net.bus_p_load is not None and net.bus_q_load is not None
    split = split and net.bus_p_gen is not None and net.bus_q_gen is not None
    s_gen = np.asarray(net.bus_p_gen) + 1j * np.asarray(net.bus_q_gen) if split else None

    if not min_x > 0.0:
        raise ValueError(f"min_x must be > 0, got {min_x}")
    Y_gs = _gs_ybus(net, Ybus, min_x)
    Y_gs.sort_indices()
    y_data, y_idx, y_indptr = Y_gs.data, Y_gs.indices, Y_gs.indptr
    diag_y = np.asarray(Y_gs.diagonal())

    for _pass in range(passes):
        s_inj = compute_sbus(net, V, voltage_dependent=True) if use_load else Sbus
        if split:
            assert s_gen is not None
            if use_load:
                s_load = s_gen - s_inj
            else:
                s_load = np.asarray(net.bus_p_load) + 1j * np.asarray(net.bus_q_load)
            as_y = (s_load.real > 0.0) & (s_load.imag > 0.0)
            vm2 = np.abs(V) ** 2
        for k in order.tolist():
            s_k, e_k = y_indptr[k], y_indptr[k + 1]
            row_dot_v = complex(np.dot(y_data[s_k:e_k], V[y_idx[s_k:e_k]]))
            others = row_dot_v - diag_y[k] * V[k]
            y_kk = diag_y[k]
            s_k_inj = complex(s_inj[k])
            if split and as_y[k]:
                assert s_gen is not None
                y_kk = y_kk + np.conj(s_load[k]) / vm2[k]
                s_k_inj = complex(s_gen[k])
            if is_pv[k]:
                q_k = (V[k] * np.conj(others + y_kk * V[k])).imag
                s_k_inj = s_k_inj.real + 1j * q_k
            v_new = (np.conj(s_k_inj / V[k]) - others) / y_kk
            if not np.isfinite(v_new):
                v_new = 1.0 + 0.0j
            if is_pv[k]:
                v_new = v_set[k] * v_new / abs(v_new) if v_new != 0 else complex(v_set[k])
            V[k] = v_new

    s_end = compute_sbus(net, V, voltage_dependent=True) if use_load else Sbus
    f = residual_vector(mismatch(Ybus, V, s_end), pv, pq)
    norm_f = residual_norm(f)
    return SolverResult(V=V, converged=norm_f < tol, iterations=passes, mismatch_max=norm_f)
