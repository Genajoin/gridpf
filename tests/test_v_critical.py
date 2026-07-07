"""Tests for the Rastr U_krit semantics (``PFInput.bus_v_critical``).

Below the per-bus critical voltage the load continues as constant impedance:
``S_load(|V|) = S_load(V_crit) · (|V|/V_crit)²`` — for both polynomial (ZIP)
and constant loads. ``bus_v_critical=None`` must stay bit-exact legacy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gridpf.algebra.sbus import compute_sbus, load_voltage_derivatives, q_load_at
from gridpf.solvers._common import resolve_use_load
from tests.test_load_voltage import _build_three_bus


V_CRIT = 0.7


def _with_crit(net, crit: float = V_CRIT):
    return replace(net, bus_v_critical=np.full(net.n_bus, crit))


class TestComputeSbusCritical:
    def test_none_is_bitexact_legacy(self) -> None:
        net = _build_three_bus(q_b=(0.0, 0.0, 1.0))
        V = np.array([1.0, 0.95, 0.55], dtype=np.complex128)
        assert net.bus_v_critical is None
        s_legacy = compute_sbus(net, V, voltage_dependent=True)
        # polynomial extrapolated below 0.7 — legacy behaviour
        assert s_legacy[2].imag == pytest.approx(-0.20 * 0.55**2, abs=1e-12)

    def test_constant_load_scales_as_const_z_below_crit(self) -> None:
        net = _with_crit(_build_three_bus())  # константная нагрузка везде
        V = np.array([1.0, 1.0, 0.35], dtype=np.complex128)
        s = compute_sbus(net, V, voltage_dependent=True)
        scale = (0.35 / V_CRIT) ** 2
        assert s[2].real == pytest.approx(-0.50 * scale, abs=1e-12)
        assert s[2].imag == pytest.approx(-0.20 * scale, abs=1e-12)
        # узел выше крита не тронут
        assert s[1].real == pytest.approx(-0.30, abs=1e-12)

    def test_poly_load_frozen_at_crit_then_scaled(self) -> None:
        net = _with_crit(_build_three_bus(q_b=(0.0, 0.0, 1.0)))
        V = np.array([1.0, 1.0, 0.35], dtype=np.complex128)
        s = compute_sbus(net, V, voltage_dependent=True)
        q_at_crit = 0.20 * V_CRIT**2
        assert s[2].imag == pytest.approx(-q_at_crit * (0.35 / V_CRIT) ** 2, abs=1e-12)

    def test_above_crit_matches_legacy(self) -> None:
        base = _build_three_bus(q_b=(0.0, 0.0, 1.0))
        net = _with_crit(base)
        V = np.array([1.0, 0.95, 0.9], dtype=np.complex128)
        np.testing.assert_allclose(
            compute_sbus(net, V, voltage_dependent=True),
            compute_sbus(base, V, voltage_dependent=True),
        )

    def test_continuity_at_crit(self) -> None:
        net = _with_crit(_build_three_bus(q_b=(0.0, 0.0, 1.0)))
        eps = 1e-9
        v_lo = np.array([1.0, 1.0, V_CRIT - eps], dtype=np.complex128)
        v_hi = np.array([1.0, 1.0, V_CRIT + eps], dtype=np.complex128)
        s_lo = compute_sbus(net, v_lo, voltage_dependent=True)
        s_hi = compute_sbus(net, v_hi, voltage_dependent=True)
        np.testing.assert_allclose(s_lo, s_hi, atol=1e-7)

    def test_q_load_at_consistent(self) -> None:
        net = _with_crit(_build_three_bus(q_b=(0.0, 0.0, 1.0)))
        V = np.array([1.0, 1.0, 0.35], dtype=np.complex128)
        q = q_load_at(net, V, voltage_dependent=True)
        q_at_crit = 0.20 * V_CRIT**2
        assert q[2] == pytest.approx(q_at_crit * (0.35 / V_CRIT) ** 2, abs=1e-12)


class TestDerivativesCritical:
    def test_below_crit_const_z_slope(self) -> None:
        net = _with_crit(_build_three_bus())  # const load
        V = np.array([1.0, 1.0, 0.35], dtype=np.complex128)
        dP, dQ = load_voltage_derivatives(net, V)
        # S(V) = S0·(V/Vc)² → dS/dV = 2·S0·V/Vc²
        assert dP[2] == pytest.approx(2.0 * 0.50 * 0.35 / V_CRIT**2, abs=1e-12)
        assert dQ[2] == pytest.approx(2.0 * 0.20 * 0.35 / V_CRIT**2, abs=1e-12)
        # выше крита у константной нагрузки производная нулевая
        assert dP[1] == 0.0

    def test_fd_matches_analytic_across_crit(self) -> None:
        """Finite differences vs analytic on both sides of the critical point."""
        net = _with_crit(_build_three_bus(p_a=(0.0, 1.0, 0.0), q_b=(0.0, 0.0, 1.0)))
        for vm2 in (0.35, 0.55, 0.75, 0.95):
            V = np.array([1.0, 1.0, vm2], dtype=np.complex128)
            dP, dQ = load_voltage_derivatives(net, V)
            eps = 1e-7
            V2 = np.array([1.0, 1.0, vm2 + eps], dtype=np.complex128)
            s1 = compute_sbus(net, V, voltage_dependent=True)
            s2 = compute_sbus(net, V2, voltage_dependent=True)
            fd_p = -(s2[2].real - s1[2].real) / eps  # S_inj = gen − load
            fd_q = -(s2[2].imag - s1[2].imag) / eps
            assert dP[2] == pytest.approx(fd_p, rel=1e-5)
            assert dQ[2] == pytest.approx(fd_q, rel=1e-5)


class TestResolveUseLoad:
    def test_crit_makes_const_load_voltage_dependent(self) -> None:
        net = _build_three_bus()  # const-P: has_voltage_dependent_load False
        assert resolve_use_load(net, True) is False
        assert resolve_use_load(_with_crit(net), True) is True
        # пользовательский off всё выключает
        assert resolve_use_load(_with_crit(net), False) is False
