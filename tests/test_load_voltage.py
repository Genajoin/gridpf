"""Тесты учёта статических характеристик нагрузки (СХН).

Проверяем:

* ``compute_sbus`` корректно применяет полиномиальную модель;
* ``load_voltage_derivatives`` даёт верные производные.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridpf import PFInput as NetworkPU
from gridpf.algebra.sbus import compute_sbus, load_voltage_derivatives
from gridpf.contract.types import BASE_MVA


def _build_three_bus(
    p_a: tuple[float, float, float] = (1.0, 0.0, 0.0),
    q_b: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> NetworkPU:
    """3-узловая сеть: SLACK ─ ветвь ─ PQ ─ ветвь ─ PQ. СХН — на втором PQ."""
    n = 3
    bus_p_a0 = np.array([1.0, 1.0, p_a[0]], dtype=np.float64)
    bus_p_a1 = np.array([0.0, 0.0, p_a[1]], dtype=np.float64)
    bus_p_a2 = np.array([0.0, 0.0, p_a[2]], dtype=np.float64)
    bus_q_b0 = np.array([1.0, 1.0, q_b[0]], dtype=np.float64)
    bus_q_b1 = np.array([0.0, 0.0, q_b[1]], dtype=np.float64)
    bus_q_b2 = np.array([0.0, 0.0, q_b[2]], dtype=np.float64)

    p_load = np.array([0.0, 0.30, 0.50], dtype=np.float64)
    q_load = np.array([0.0, 0.10, 0.20], dtype=np.float64)
    p_gen = np.zeros(n, dtype=np.float64)
    q_gen = np.zeros(n, dtype=np.float64)

    return NetworkPU(
        n_bus=n,
        n_branch=2,
        bus_ids=np.array([1, 2, 3], dtype=np.int64),
        bus_vn_kv=np.full(n, 110.0),
        bus_type=np.array([2, 0, 0], dtype=np.int8),
        slack_idx=0,
        branch_ids=np.array([10, 11], dtype=np.int64),
        from_idx=np.array([0, 1], dtype=np.int64),
        to_idx=np.array([1, 2], dtype=np.int64),
        branch_r=np.array([0.01, 0.01]),
        branch_x=np.array([0.1, 0.1]),
        branch_g=np.zeros(2),
        branch_b=np.zeros(2),
        branch_g_from=np.zeros(2),
        branch_b_from=np.zeros(2),
        branch_g_to=np.zeros(2),
        branch_b_to=np.zeros(2),
        tap_ratio=np.ones(2),
        phase_shift=np.zeros(2),
        bus_g_shunt=np.zeros(n),
        bus_b_shunt=np.zeros(n),
        bus_p_injection=p_gen - p_load,
        bus_q_injection=q_gen - q_load,
        bus_p_gen=p_gen,
        bus_q_gen=q_gen,
        bus_p_load=p_load,
        bus_q_load=q_load,
        bus_p_a0=bus_p_a0,
        bus_p_a1=bus_p_a1,
        bus_p_a2=bus_p_a2,
        bus_q_b0=bus_q_b0,
        bus_q_b1=bus_q_b1,
        bus_q_b2=bus_q_b2,
        base_mva=BASE_MVA,
    )


class TestComputeSbus:
    def test_constant_load_matches_baseline(self) -> None:
        """При a0=1, остальные 0 (default) compute_sbus = build_sbus."""
        from gridpf.algebra.sbus import build_sbus

        net = _build_three_bus()
        V = np.array([1.0, 0.95, 0.90], dtype=np.complex128)
        s_const = build_sbus(net)
        s_v = compute_sbus(net, V, voltage_dependent=False)
        np.testing.assert_allclose(s_v, s_const)

    def test_zip_load_decreases_with_voltage_drop(self) -> None:
        """Чисто квадратичная Q-нагрузка: при |V|=0.9, Q_load = 0.81·Q0 → Q_inj растёт."""
        net = _build_three_bus(q_b=(0.0, 0.0, 1.0))  # Q(V) = Q0·V²
        V = np.array([1.0, 1.0, 0.9], dtype=np.complex128)
        s = compute_sbus(net, V, voltage_dependent=True)
        # На узле 2: Q_load_v = 0.20 · 0.81 = 0.162 → Q_inj = -0.162
        assert s[2].imag == pytest.approx(-0.162, abs=1e-9)
        # На остальных узлах СХН тривиальна → как baseline
        assert s[0].imag == pytest.approx(0.0)
        assert s[1].imag == pytest.approx(-0.10)

    def test_linear_p_load(self) -> None:
        """Линейная P: a1=1 → P_load(V) = P0·V."""
        net = _build_three_bus(p_a=(0.0, 1.0, 0.0))
        V = np.array([1.0, 1.0, 0.5], dtype=np.complex128)
        s = compute_sbus(net, V, voltage_dependent=True)
        # На узле 2: P_load = 0.50 · 0.5 = 0.25
        assert s[2].real == pytest.approx(-0.25, abs=1e-9)


class TestLoadVoltageDerivatives:
    def test_constant_load_zero_derivatives(self) -> None:
        net = _build_three_bus()  # default a0=1
        V = np.ones(3, dtype=np.complex128)
        dP, dQ = load_voltage_derivatives(net, V)
        np.testing.assert_array_equal(dP, 0.0)
        np.testing.assert_array_equal(dQ, 0.0)

    def test_quadratic_q_derivative(self) -> None:
        """Q(V) = Q0·V² → dQ/dV = 2·Q0·V. При V=1, Q0=0.20 → dQ/dV = 0.4."""
        net = _build_three_bus(q_b=(0.0, 0.0, 1.0))
        V = np.ones(3, dtype=np.complex128)
        dP, dQ = load_voltage_derivatives(net, V)
        np.testing.assert_array_equal(dP, 0.0)
        assert dQ[2] == pytest.approx(0.4, abs=1e-12)
        assert dQ[0] == 0.0
        assert dQ[1] == 0.0
