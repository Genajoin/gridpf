"""Transformer branch shunts: the "from" shunt sits at the "from" node.

Branch layout (see ``gridpf.algebra.ybus``): the series impedance at the "from"
end, the ideal transformer ``t : 1`` at the "to" end, the whole "from" shunt at
the "from" node in the "from"-side base (before the ideal transformer,
independent of the tap), the "to" shunt at the "to" node::

    from ──┬── r + j·x ──( t : 1 )──┬── to
         Yc_from                   Yc_to

The network below has transformers with off-nominal taps, a phase shift and
non-zero shunts on both ends. The solved branch flows are compared with a
direct evaluation of this circuit, and the nodal balance is checked against
the injections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from gridpf import BASE_MVA, PFInput, PFOptions, build_ybus, solve


def _network(build_net: Callable[..., PFInput], *, tap_scale: float = 1.0) -> PFInput:
    """Slack(0) → PQ(1) → PQ(2) through two transformers, closed by a line 0–2.

    The second transformer starts at a PQ bus, so its "from" shunt enters the
    nodal balance that the solver enforces.
    """
    net = build_net(
        bus_type=[2, 0, 0],
        edges=[(0, 1, 0.004, 0.08), (1, 2, 0.006, 0.11), (0, 2, 0.01, 0.04)],
        p_inj=[0.0, -0.6, -0.4],
        q_inj=[0.0, -0.25, -0.15],
    )
    return replace(
        net,
        tap_ratio=np.array([1.06 * tap_scale, 0.94 * tap_scale, 1.0]),
        phase_shift=np.array([0.03, 0.0, 0.0]),
        branch_b=np.array([0.0, 0.0, 0.02]),
        branch_g_from=np.array([0.002, 0.003, 0.0]),
        branch_b_from=np.array([-0.015, -0.025, 0.0]),
        branch_g_to=np.array([0.001, 0.0, 0.0]),
        branch_b_to=np.array([-0.004, 0.0, 0.0]),
    )


def _circuit_flows(net: PFInput, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Branch flows (MVA, into the branch at each end) evaluated on the circuit."""
    vf = V[net.from_idx]
    vt = V[net.to_idx]
    t = net.tap_ratio * np.exp(1j * net.phase_shift)
    ysf = 1.0 / (net.branch_r + 1j * net.branch_x)
    yc_half = 0.5 * (net.branch_g + 1j * net.branch_b)
    yc_from = net.branch_g_from + 1j * net.branch_b_from + yc_half
    yc_to = net.branch_g_to + 1j * net.branch_b_to + yc_half

    # Series current from the "from" node towards the ideal transformer, whose
    # primary side is at t·V_to; the secondary current is conj(t) times larger.
    i_series = (vf - t * vt) * ysf
    i_from = i_series + yc_from * vf
    i_to = -np.conj(t) * i_series + yc_to * vt
    return vf * np.conj(i_from) * BASE_MVA, vt * np.conj(i_to) * BASE_MVA


@pytest.fixture
def net(build_net: Callable[..., PFInput]) -> PFInput:
    return _network(build_net)


def test_solved_flows_match_circuit(net: PFInput) -> None:
    res = solve(net, PFOptions(method="nr"))
    assert res.converged

    s_from, s_to = _circuit_flows(net, res.V)
    np.testing.assert_allclose(res.S_from, s_from, rtol=0, atol=1e-9)
    np.testing.assert_allclose(res.S_to, s_to, rtol=0, atol=1e-9)
    np.testing.assert_allclose(res.branch_loss_p, np.real(s_from + s_to), rtol=0, atol=1e-9)


def test_nodal_balance_matches_injections(net: PFInput) -> None:
    """Flows leaving each bus sum to its injection (Jacobian/Y-bus consistency)."""
    res = solve(net, PFOptions(method="nr", tol=1e-12))
    assert res.converged

    s_from, s_to = _circuit_flows(net, res.V)
    s_out = np.zeros(net.n_bus, dtype=complex)
    np.add.at(s_out, net.from_idx, s_from)
    np.add.at(s_out, net.to_idx, s_to)
    s_inj = (net.bus_p_injection + 1j * net.bus_q_injection) * BASE_MVA
    pq = net.bus_type == 0
    np.testing.assert_allclose(s_out[pq], s_inj[pq], rtol=0, atol=1e-8)


def test_from_shunt_is_independent_of_tap(build_net: Callable[..., PFInput]) -> None:
    """``Yff - Ysf`` equals ``Yc_from`` for any tap; ``Ytt - Ysf·|t|²`` equals ``Yc_to``."""
    for tap_scale in (0.9, 1.0, 1.1):
        net = _network(build_net, tap_scale=tap_scale)
        _, yf, yt = build_ybus(net)
        rows = np.arange(net.n_branch)
        ysf = 1.0 / (net.branch_r + 1j * net.branch_x)
        yc_half = 0.5 * (net.branch_g + 1j * net.branch_b)
        yff = np.asarray(yf[rows, net.from_idx]).ravel()
        ytt = np.asarray(yt[rows, net.to_idx]).ravel()
        np.testing.assert_allclose(
            yff - ysf, net.branch_g_from + 1j * net.branch_b_from + yc_half, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            ytt - ysf * net.tap_ratio**2,
            net.branch_g_to + 1j * net.branch_b_to + yc_half,
            rtol=0,
            atol=1e-12,
        )
