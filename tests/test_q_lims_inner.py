"""Tests for Q-limit switching inside the Newton iteration and ``bus_q_lock``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridpf import PFInput, PFOptions, solve
from gridpf.algebra.sbus import q_load_at
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.validate import PFContractValidationError, validate_input
from gridpf.solvers.q_lims import InnerQLimits


INNER = PFOptions(enforce_q_lims=True, q_lim_mode="inner")


def _gen_net(
    build_net: Callable[..., PFInput], q_max: float = 0.05, v_set: float = 1.05
) -> PFInput:
    # slack(0) — PV(1, |V|=v_set, gives 0.3 p.u.) — PQ(2, load 0.5 + j0.3).
    return build_net(
        bus_type=[2, 1, 0],
        edges=[(0, 1, 0.01, 0.05), (1, 2, 0.02, 0.06)],
        p_inj=[0.0, 0.3, -0.5],
        q_inj=[0.0, 0.0, -0.3],
        bus_v_set=[1.0, v_set, np.nan],
        bus_va_set=[0.0, np.nan, np.nan],
        q_min=[np.nan, -q_max, np.nan],
        q_max=[np.nan, q_max, np.nan],
    )


def _q_gen(net: PFInput, V: np.ndarray) -> np.ndarray:
    ybus, _, _ = build_ybus(net)
    return np.asarray((V * np.conj(ybus @ V)).imag + q_load_at(net, V))


class TestEngine:
    def test_default_mode_is_outer(self) -> None:
        assert PFOptions().q_lim_mode == "outer"

    def test_pins_generator_at_qmax(self, build_net: Callable[..., PFInput]) -> None:
        net = _gen_net(build_net)
        free = solve(net, PFOptions())
        assert _q_gen(net, free.V)[1] > 0.05  # the limit is binding
        res = solve(net, INNER)
        assert res.converged
        assert res.q_lim_swaps >= 1
        assert res.q_violations == 0
        assert _q_gen(net, res.V)[1] == pytest.approx(0.05, abs=1e-7)
        assert abs(res.V[1]) < 1.05

    def test_inner_matches_outer_final_state(self, build_net: Callable[..., PFInput]) -> None:
        net = _gen_net(build_net)
        outer = solve(net, PFOptions(enforce_q_lims=True))
        inner = solve(net, INNER)
        assert outer.converged and inner.converged
        np.testing.assert_allclose(inner.V, outer.V, atol=1e-7)

    def test_non_binding_limit_untouched(self, build_net: Callable[..., PFInput]) -> None:
        net = _gen_net(build_net, q_max=5.0)
        free = solve(net, PFOptions())
        res = solve(net, INNER)
        assert res.converged and res.q_lim_swaps == 0
        np.testing.assert_array_equal(res.V, free.V)


class TestInitialLock:
    def test_wrong_lock_returns_to_pv(self, build_net: Callable[..., PFInput]) -> None:
        # Generator flagged at Qmax although it only needs a fraction of it:
        # |V| rises above V_set, the inner mode returns the bus to PV.
        net = replace(_gen_net(build_net, q_max=2.0), bus_q_lock=np.array([0, 1, 0], np.int8))
        res = solve(net, INNER)
        assert res.converged
        assert abs(res.V[1]) == pytest.approx(1.05, abs=1e-9)
        assert _q_gen(net, res.V)[1] < 2.0

    def test_outer_without_return_keeps_lock(self, build_net: Callable[..., PFInput]) -> None:
        net = replace(_gen_net(build_net, q_max=2.0), bus_q_lock=np.array([0, 1, 0], np.int8))
        res = solve(net, PFOptions(enforce_q_lims=True))
        assert res.converged
        assert _q_gen(net, res.V)[1] == pytest.approx(2.0, abs=1e-7)

    def test_lock_ignored_without_enforcement(self, build_net: Callable[..., PFInput]) -> None:
        base = _gen_net(build_net, q_max=2.0)
        net = replace(base, bus_q_lock=np.array([0, 1, 0], np.int8))
        np.testing.assert_array_equal(solve(net, PFOptions()).V, solve(base, PFOptions()).V)

    def test_zero_lock_bit_for_bit(self, build_net: Callable[..., PFInput]) -> None:
        base = _gen_net(build_net)
        net = replace(base, bus_q_lock=np.zeros(3, np.int8))
        opts = PFOptions(enforce_q_lims=True)
        np.testing.assert_array_equal(solve(net, opts).V, solve(base, opts).V)


class TestValidation:
    def test_valid_lock(self, build_net: Callable[..., PFInput]) -> None:
        net = replace(_gen_net(build_net), bus_q_lock=np.array([0, -1, 0], np.int8))
        assert validate_input(net).ok

    @pytest.mark.parametrize(
        "lock",
        [
            [0, 2, 0],  # bad value
            [0, 0, 1],  # not a PV bus
            [0, 1],  # wrong length
        ],
    )
    def test_invalid_lock(self, build_net: Callable[..., PFInput], lock: list[int]) -> None:
        net = replace(_gen_net(build_net), bus_q_lock=np.array(lock, np.int8))
        with pytest.raises(PFContractValidationError, match="bus_q_lock"):
            validate_input(net, strict=True)

    def test_lock_requires_limit(self, build_net: Callable[..., PFInput]) -> None:
        net = replace(
            _gen_net(build_net),
            bus_q_max=np.array([np.nan, np.nan, np.nan]),
            bus_q_lock=np.array([0, 1, 0], np.int8),
        )
        with pytest.raises(PFContractValidationError, match="bus_q_max"):
            validate_input(net, strict=True)


def _controller(
    n: int, pv: list[int], side: list[int], q_gen: list[float], **kw: object
) -> InnerQLimits:
    """Controller on a dummy network with a stubbed ``Q_gen``."""
    ctrl = InnerQLimits(
        csr_matrix((n, n), dtype=np.complex128),
        pv=np.array(pv, dtype=np.int64),
        pq=np.array([k for k in range(n) if k not in pv], dtype=np.int64),
        Sbus=np.zeros(n, dtype=np.complex128),
        net=_DummyNet(n),  # type: ignore[arg-type]
        q_min=np.full(n, -1.0),
        q_max=np.full(n, 1.0),
        v_set=np.ones(n),
        side=np.array(side, dtype=np.int8),
        pv_original=np.arange(n, dtype=np.int64),
        q_lim_tol=0.0,
        v_deadband_rel=float(kw.get("deadband", 1e-5)),  # type: ignore[arg-type]
        max_returns=int(kw.get("max_returns", 4)),  # type: ignore[call-overload]
        voltage_dependent_load=False,
    )
    q = np.array(q_gen, dtype=np.float64)
    ctrl._q_gen = lambda V: q  # type: ignore[method-assign]
    return ctrl


class _DummyNet:
    """Just enough of ``PFInput`` for :func:`q_load_at` (no load data)."""

    def __init__(self, n: int) -> None:
        self.n_bus = n
        self.bus_q_load = None
        self.bus_q_gen = None


class TestController:
    def test_group_one_has_priority(self) -> None:
        # bus 0 over Qmax (group 1), bus 1 under Qmin (group 2).
        ctrl = _controller(2, pv=[0, 1], side=[0, 0], q_gen=[1.5, -1.5])
        changed, released = ctrl.convergence_switch(np.ones(2, dtype=np.complex128))
        assert changed and released.size == 0
        assert ctrl.side.tolist() == [1, 0]
        changed, _ = ctrl.convergence_switch(np.ones(2, dtype=np.complex128))
        assert changed and ctrl.side.tolist() == [1, -1]

    def test_qmin_return_joins_group_one(self) -> None:
        # bus 0 over Qmax, bus 1 pinned at Qmin with |V| below V_set → both in group 1.
        ctrl = _controller(2, pv=[0], side=[0, -1], q_gen=[1.5, -1.0])
        V = np.array([1.0, 0.99], dtype=np.complex128)
        changed, released = ctrl.convergence_switch(V)
        assert changed and released.tolist() == [1]
        assert ctrl.side.tolist() == [1, 0]
        assert ctrl.pv.tolist() == [1] and ctrl.pq.tolist() == [0]

    def test_relative_deadband(self) -> None:
        ctrl = _controller(1, pv=[], side=[1], q_gen=[1.0])
        changed, _ = ctrl.convergence_switch(np.array([1.0 + 5e-6], dtype=np.complex128))
        assert not changed
        changed, released = ctrl.convergence_switch(np.array([1.0 + 2e-5], dtype=np.complex128))
        assert changed and released.tolist() == [0]

    def test_return_cap(self) -> None:
        ctrl = _controller(1, pv=[], side=[1], q_gen=[1.0], max_returns=1)
        V_hi = np.array([1.01], dtype=np.complex128)
        assert ctrl.convergence_switch(V_hi)[0]  # first return
        ctrl._q_gen = lambda V: np.array([1.5])  # type: ignore[method-assign]
        assert ctrl.convergence_switch(V_hi)[0]  # pinned again
        assert not ctrl.convergence_switch(V_hi)[0]  # cap reached: stays pinned
        assert ctrl.side.tolist() == [1]

    def test_step_check_thresholds(self) -> None:
        # bus 0 pinned in this run (1 % rule applies), bus 1 pinned at start.
        ctrl = _controller(2, pv=[0], side=[0, 1], q_gen=[1.5, 1.0])
        ctrl.convergence_switch(np.ones(2, dtype=np.complex128))
        released, undo = ctrl.step_check(np.array([1.02, 1.02], dtype=np.complex128))
        assert released.tolist() == [0] and not undo
        released, undo = ctrl.step_check(np.array([1.0, 1.12], dtype=np.complex128))
        assert released.tolist() == [1] and undo
