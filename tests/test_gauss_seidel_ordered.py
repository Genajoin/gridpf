"""Tests for the ordered Gauss-Seidel warm start, the GS guard and the warm-start check."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from gridpf import PFInput, PFOptions, solve
from gridpf._engine import _Classification, _gs_made_worse
from gridpf.algebra.sbus import build_sbus, classify_buses, compute_sbus
from gridpf.algebra.ybus import build_ybus
from gridpf.solvers._common import SolverResult
from gridpf.solvers.gauss_seidel_ordered import gauss_seidel_ordered, traversal_order


def _net(build_net: Callable[..., PFInput]) -> PFInput:
    # slack(0) ─ 1 ─(breaker)─ 2 ─ 3 (PV) ; bus 4 hangs off bus 1 through a long line.
    return build_net(
        bus_type=[2, 0, 0, 1, 0],
        edges=[
            (0, 1, 0.01, 0.05),
            (1, 2, 0.0, 1e-4),
            (2, 3, 0.02, 0.06),
            (1, 4, 0.05, 0.3),
        ],
        p_inj=[0.0, -0.4, -0.2, 0.3, -0.1],
        q_inj=[0.0, -0.2, -0.1, 0.0, -0.05],
        bus_v_set=[1.0, np.nan, np.nan, 1.02, np.nan],
        bus_va_set=[0.0, np.nan, np.nan, np.nan, np.nan],
        schn={"bus_q_b2": [0.0, 1.0, 0.0, 0.0, 0.0]},
    )


class TestOrder:
    def test_breaker_neighbours_follow_each_other(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        ref, _, _ = classify_buses(net.bus_type)
        assert traversal_order(net, ref).tolist() == [1, 2, 3, 4]

    def test_unreachable_buses_are_skipped(self, build_net: Callable[..., PFInput]) -> None:
        net = build_net(
            bus_type=[2, 0, 0],
            edges=[(0, 1, 0.01, 0.05)],
            p_inj=[0.0, -0.1, 0.0],
            q_inj=[0.0, 0.0, 0.0],
        )
        ref, _, _ = classify_buses(net.bus_type)
        assert traversal_order(net, ref).tolist() == [1]

    def test_transformer_refers_the_impedance(self, build_net: Callable[..., PFInput]) -> None:
        # Two paths from the slack; the ratio on the first branch scales the
        # rest of its path by s = 1/|t|, which makes it shorter.
        net = build_net(
            bus_type=[2, 0, 0, 0],
            edges=[(0, 1, 0.0, 0.10), (1, 3, 0.0, 0.10), (0, 2, 0.0, 0.11), (2, 3, 0.0, 0.11)],
            p_inj=[0.0, 0.0, 0.0, -0.1],
            q_inj=[0.0, 0.0, 0.0, 0.0],
        )
        ref, _, _ = classify_buses(net.bus_type)
        assert traversal_order(net, ref).tolist() == [1, 2, 3]
        tap = np.array([0.5, 1.0, 1.0, 1.0])
        net_t = replace(net, tap_ratio=tap)
        # bus 3 via bus 1: 0.10 + (1/0.5)²·0.10 = 0.50 > 0.22 via bus 2.
        assert traversal_order(net_t, ref).tolist() == [1, 2, 3]


class TestSolver:
    def test_solution_is_a_fixed_point(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        sol = solve(net, PFOptions(method="nr", tol=1e-11))
        assert sol.converged
        ybus, _, _ = build_ybus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        sbus = compute_sbus(net, sol.V)
        res = gauss_seidel_ordered(
            ybus, sbus, sol.V, ref, pv, pq, net=net, passes=2, tol=1e-6, voltage_dependent_load=True
        )
        np.testing.assert_allclose(res.V, sol.V, atol=1e-10)

    def test_many_passes_approach_the_solution(self, build_net: Callable[..., PFInput]) -> None:
        # No breaker here: Gauss-Seidel itself converges slowly across a
        # near-zero impedance, ordered or not.
        net = replace(_net(build_net), branch_x=np.array([0.05, 0.05, 0.06, 0.3]))
        sol = solve(net, PFOptions(method="nr", tol=1e-11))
        res = solve(net, PFOptions(method="gs", gs_mode="ordered", max_iter_gs=400, tol=1e-6))
        assert res.iterations_gs == 400
        np.testing.assert_allclose(res.V, sol.V, atol=1e-5)

    def test_ordered_warm_start_then_newton(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        ref = solve(net, PFOptions(method="nr"))
        res = solve(net, PFOptions(gs_mode="ordered", max_iter_gs=4))
        assert res.converged and res.iterations_gs == 4
        np.testing.assert_allclose(res.V, ref.V, atol=1e-8)

    def test_negative_passes_rejected(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        ybus, _, _ = build_ybus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        with pytest.raises(ValueError, match="passes"):
            gauss_seidel_ordered(
                ybus, build_sbus(net), np.ones(5, complex), ref, pv, pq, net=net, passes=-1, tol=1.0
            )


class TestGuards:
    def test_defaults(self) -> None:
        opts = PFOptions()
        assert opts.gs_mode == "classic" and not opts.gs_guard and opts.init_v_max_mismatch is None

    def test_guard_flags_worse_and_non_finite(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        ybus, _, _ = build_ybus(net)
        _, pv, pq = classify_buses(net.bus_type)
        V0 = np.ones(5, dtype=np.complex128)
        cls = _Classification(
            pv=pv, pq=pq, Sbus=build_sbus(net), locked_lim=np.full(5, np.nan), net=net
        )
        opts = PFOptions(gs_guard=True)
        worse = SolverResult(V=V0 * 3.0, converged=False, iterations=4, mismatch_max=1e9)
        assert _gs_made_worse(worse, V0, ybus, cls, opts, False)
        nan = SolverResult(V=V0 * np.nan, converged=False, iterations=4, mismatch_max=np.nan)
        assert _gs_made_worse(nan, V0, ybus, cls, opts, False)
        better = SolverResult(V=V0, converged=False, iterations=4, mismatch_max=0.0)
        assert not _gs_made_worse(better, V0, ybus, cls, opts, False)
        outside = SolverResult(V=V0 * 2.5, converged=False, iterations=4, mismatch_max=0.0)
        assert _gs_made_worse(
            outside, V0, ybus, cls, replace(opts, step_v_corridor=(0.5, 2.0)), False
        )

    def test_guard_keeps_good_warm_start(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        a = solve(net, PFOptions())
        b = solve(net, PFOptions(gs_guard=True))
        np.testing.assert_array_equal(a.V, b.V)

    def test_bad_init_v_falls_back_to_flat(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        flat = solve(net, PFOptions())
        bad = np.full(5, 0.3 + 0.5j)
        res = solve(net, PFOptions(init_v_max_mismatch=1.0), init_v=bad)
        np.testing.assert_array_equal(res.V, flat.V)

    def test_good_init_v_is_kept(self, build_net: Callable[..., PFInput]) -> None:
        net = _net(build_net)
        sol = solve(net, PFOptions())
        a = solve(net, PFOptions(), init_v=sol.V)
        b = solve(net, PFOptions(init_v_max_mismatch=1.0), init_v=sol.V)
        np.testing.assert_array_equal(a.V, b.V)


class TestSeriesCapacitance:
    def test_negative_x_branch_shorted_for_gs_only(self, build_net: Callable[..., PFInput]) -> None:
        from gridpf.solvers.gauss_seidel_ordered import _gs_ybus

        net = replace(_net(build_net), branch_x=np.array([0.05, -0.0015, 0.06, 0.3]))
        ybus, _, _ = build_ybus(net)
        y_gs = _gs_ybus(net, ybus)
        y_old = 1.0 / (0.0 - 0.0015j)
        y_new = y_old.real - 1j * 110.0 / 0.022
        assert y_gs[1, 2] == pytest.approx(-y_new)
        assert ybus[1, 2] == pytest.approx(-y_old)  # the Newton Ybus is untouched
        # other branches unchanged
        assert y_gs[0, 1] == pytest.approx(ybus[0, 1])

    def test_no_negative_x_returns_same_ybus(self, build_net: Callable[..., PFInput]) -> None:
        from gridpf.solvers.gauss_seidel_ordered import _gs_ybus

        net = _net(build_net)
        ybus, _, _ = build_ybus(net)
        assert _gs_ybus(net, ybus) is ybus
