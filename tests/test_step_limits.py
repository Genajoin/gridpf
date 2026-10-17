"""Tests for the Newton step-size limits and the post-step corridor."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridpf import PFInput, PFOptions, solve
from gridpf.algebra.sbus import build_sbus, classify_buses
from gridpf.algebra.ybus import build_ybus
from gridpf.solvers.newton_raphson import newton_raphson
from gridpf.solvers.step_limits import (
    StepLimits,
    branch_pairs,
    corridor_correction,
    expand_increment,
    step_scale,
)


def _chain(build_net: Callable[..., PFInput], load: float = 0.5) -> PFInput:
    return build_net(
        bus_type=[2, 1, 0, 0],
        edges=[(0, 1, 0.01, 0.05), (1, 2, 0.02, 0.08), (2, 3, 0.02, 0.08)],
        p_inj=[0.0, 0.3, -load, -load],
        q_inj=[0.0, 0.0, -0.2, -0.2],
        bus_v_set=[1.0, 1.03, np.nan, np.nan],
        bus_va_set=[0.0, np.nan, np.nan, np.nan],
    )


def _nr(net: PFInput, limits: StepLimits | None, **kw: float) -> tuple[np.ndarray, int, bool]:
    ybus, _, _ = build_ybus(net)
    sbus = build_sbus(net)
    ref, pv, pq = classify_buses(net.bus_type)
    V0 = np.ones(net.n_bus, dtype=np.complex128)
    V0[pv] = 1.03
    res = newton_raphson(ybus, sbus, V0, ref, pv, pq, step_limits=limits, **kw)  # type: ignore[arg-type]
    return res.V, res.iterations, res.converged


class TestStepScale:
    def test_no_limits_is_one(self) -> None:
        Vm = np.ones(3)
        assert step_scale(Vm, np.full(3, 5.0), np.full(3, 5.0), StepLimits(), None) == 1.0

    def test_modulus_limit_weighted_by_voltage(self) -> None:
        Vm = np.array([1.0, 0.5])
        dvm = np.array([0.0, 0.1])  # 0.1 / 0.5 / 0.5 = 0.4
        alpha = step_scale(Vm, np.zeros(2), dvm, StepLimits(max_dvm=0.2), None)
        assert alpha == pytest.approx(0.5)

    def test_angle_and_branch_limits_take_minimum(self) -> None:
        Vm = np.ones(3)
        dva = np.array([0.0, 0.4, -0.4])
        pairs = (np.array([0, 1]), np.array([1, 2]))
        limits = StepLimits(max_dva=0.8, max_branch_dva=0.2)
        # branch (1, 2): |0.4 − (−0.4)| = 0.8 → α = 0.25; node angle 0.4 is within 0.8.
        assert step_scale(Vm, dva, np.zeros(3), limits, pairs) == pytest.approx(0.25)

    def test_expand_increment(self) -> None:
        pv = np.array([1])
        pq = np.array([2, 3])
        dva, dvm = expand_increment(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 4, pv, pq)
        np.testing.assert_array_equal(dva, [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(dvm, [0.0, 0.0, 0.4, 0.5])


class TestCorridor:
    def test_upper_violator_lands_on_boundary(self) -> None:
        Vm_old = np.array([1.0, 1.8, 1.0])
        Vm_new = np.array([1.0, 2.2, 1.1])
        Va = np.zeros(3)
        a = corridor_correction(Vm_old, Va, Vm_new, Va, StepLimits(v_corridor=(0.5, 2.0)), None)
        assert Vm_new[1] + a * (Vm_new[1] - Vm_old[1]) == pytest.approx(2.0)

    def test_lower_violator_lands_on_boundary(self) -> None:
        Vm_old = np.array([0.7, 1.0])
        Vm_new = np.array([0.4, 1.0])
        Va = np.zeros(2)
        a = corridor_correction(Vm_old, Va, Vm_new, Va, StepLimits(v_corridor=(0.5, 2.0)), None)
        assert Vm_new[0] + a * (Vm_new[0] - Vm_old[0]) == pytest.approx(0.5)

    def test_inward_motion_is_not_a_violation(self) -> None:
        Vm_old = np.array([0.3])
        Vm_new = np.array([0.4])
        Va = np.zeros(1)
        assert (
            corridor_correction(Vm_old, Va, Vm_new, Va, StepLimits(v_corridor=(0.5, 2.0)), None)
            == 0.0
        )

    def test_already_outside_is_left_to_newton(self) -> None:
        # Already below the corridor (incl. a transient negative modulus) and
        # moving further out: rolling it onto the boundary would need a < −1.
        Vm_old = np.array([0.45, -0.47])
        Vm_new = np.array([0.30, -0.50])
        Va = np.zeros(2)
        assert (
            corridor_correction(Vm_old, Va, Vm_new, Va, StepLimits(v_corridor=(0.5, 2.0)), None)
            == 0.0
        )

    def test_branch_angle(self) -> None:
        Vm = np.ones(2)
        pairs = (np.array([0]), np.array([1]))
        Va_old = np.array([0.0, -1.0])
        Va_new = np.array([0.0, -2.0])
        limits = StepLimits(max_branch_angle=np.pi / 2)
        a = corridor_correction(Vm, Va_old, Vm, Va_new, limits, pairs)
        assert 2.0 + a * 1.0 == pytest.approx(np.pi / 2)

    def test_branch_pairs_from_ybus(self) -> None:
        ybus = csr_matrix(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.complex128))
        i, j = branch_pairs(ybus)
        assert sorted(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (1, 2)]


class TestValidation:
    @pytest.mark.parametrize("field", ["max_dvm", "max_dva", "max_branch_dva", "max_branch_angle"])
    def test_non_positive_limit_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            StepLimits(**{field: 0.0})

    def test_bad_corridor_rejected(self) -> None:
        with pytest.raises(ValueError, match="step_v_corridor"):
            StepLimits(v_corridor=(1.5, 0.5))


class TestNewton:
    def test_inactive_limits_bit_for_bit(self, build_net: Callable[..., PFInput]) -> None:
        net = _chain(build_net)
        V_ref, it_ref, _ = _nr(net, None)
        V_off, it_off, _ = _nr(net, StepLimits())
        np.testing.assert_array_equal(V_off, V_ref)
        assert it_off == it_ref

    def test_tight_modulus_limit_same_solution_more_iterations(
        self, build_net: Callable[..., PFInput]
    ) -> None:
        net = _chain(build_net)
        V_ref, it_ref, ok_ref = _nr(net, None)
        V_lim, it_lim, ok_lim = _nr(net, StepLimits(max_dvm=0.01), max_iter=100)
        assert ok_ref and ok_lim
        assert it_lim > it_ref
        np.testing.assert_allclose(V_lim, V_ref, atol=1e-8)

    def test_corridor_inside_solution_converges(self, build_net: Callable[..., PFInput]) -> None:
        net = _chain(build_net)
        V_ref, _, _ = _nr(net, None)
        limits = StepLimits(
            max_dvm=0.2,
            max_dva=np.pi / 2,
            max_branch_dva=np.pi / 6,
            v_corridor=(0.5, 2.0),
            max_branch_angle=np.pi / 2,
        )
        V_lim, _, ok = _nr(net, limits)
        assert ok
        np.testing.assert_allclose(V_lim, V_ref, atol=1e-8)


class TestEngine:
    def test_default_options_bit_for_bit(self, build_net: Callable[..., PFInput]) -> None:
        net = _chain(build_net)
        a = solve(net, PFOptions())
        b = solve(
            net,
            PFOptions(
                step_max_dvm=None,
                step_max_dva=None,
                step_max_branch_dva=None,
                step_v_corridor=None,
                step_max_branch_angle=None,
            ),
        )
        np.testing.assert_array_equal(a.V, b.V)

    def test_options_reach_the_solver(self, build_net: Callable[..., PFInput]) -> None:
        net = _chain(build_net)
        ref = solve(net, PFOptions(method="nr"))
        lim = solve(net, PFOptions(method="nr", step_max_dvm=0.01, max_iter_nr=100))
        assert ref.converged and lim.converged
        assert lim.iterations_nr > ref.iterations_nr
        np.testing.assert_allclose(lim.V, ref.V, atol=1e-8)
