"""Тесты Gauss-Seidel итератора."""

from __future__ import annotations

import numpy as np
import pytest

from gridpf import PFInput as NetworkPU
from gridpf.algebra.sbus import build_sbus, classify_buses
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.types import BASE_MVA
from gridpf.solvers.gauss_seidel import gauss_seidel


def _two_bus_pq(p_load_mw: float = 50.0, q_load_mvar: float = 20.0) -> NetworkPU:
    """Slack ↔ PQ с одной чисто реактивной линией.

    bus 0 — slack (V=1.0∠0); bus 1 — PQ с нагрузкой.
    """
    n = 2
    return NetworkPU(
        n_bus=n,
        n_branch=1,
        bus_ids=np.array([1, 2], dtype=np.int64),
        bus_vn_kv=np.array([110.0, 110.0]),
        bus_type=np.array([2, 0], dtype=np.int8),
        slack_idx=0,
        branch_ids=np.array([10], dtype=np.int64),
        from_idx=np.array([0], dtype=np.int64),
        to_idx=np.array([1], dtype=np.int64),
        branch_r=np.array([0.01]),
        branch_x=np.array([0.1]),
        branch_g=np.array([0.0]),
        branch_b=np.array([0.0]),
        branch_g_from=np.array([0.0]),
        branch_b_from=np.array([0.0]),
        branch_g_to=np.array([0.0]),
        branch_b_to=np.array([0.0]),
        tap_ratio=np.array([1.0]),
        phase_shift=np.array([0.0]),
        bus_g_shunt=np.zeros(n),
        bus_b_shunt=np.zeros(n),
        # Нагрузка задаётся отрицательной инъекцией.
        bus_p_injection=np.array([0.0, -p_load_mw / BASE_MVA]),
        bus_q_injection=np.array([0.0, -q_load_mvar / BASE_MVA]),
        base_mva=BASE_MVA,
    )


class TestGaussSeidelBasic:
    def test_flat_start_already_converged(self) -> None:
        """Без нагрузок и шунтов flat-старт уже идеален."""
        net = _two_bus_pq(p_load_mw=0.0, q_load_mvar=0.0)
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        v0 = np.ones(net.n_bus, dtype=np.complex128)
        res = gauss_seidel(ybus, sbus, v0, ref, pv, pq, tol=1e-10, max_iter=20)
        assert res.converged
        assert res.iterations == 0

    def test_two_bus_load_converges(self) -> None:
        net = _two_bus_pq(p_load_mw=50.0, q_load_mvar=20.0)
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        v0 = np.ones(net.n_bus, dtype=np.complex128)
        res = gauss_seidel(ybus, sbus, v0, ref, pv, pq, tol=1e-8, max_iter=200)
        assert res.converged, f"GS не сошёлся за 200 итераций; mismatch={res.mismatch_max:.2e}"
        # Slack — без изменений
        np.testing.assert_allclose(res.V[0], 1.0 + 0j, atol=1e-12)
        # Под нагрузкой |V| на втором узле должно просесть < 1
        assert abs(res.V[1]) < 1.0
        assert abs(res.V[1]) > 0.9
        # Финальный небаланс — V·conj(Y·V) = S
        mis = res.V * np.conj(ybus @ res.V) - sbus
        # На slack — большой (компенсируется), на PQ — около нуля.
        assert abs(mis[1]) < 1e-8


class TestGaussSeidelEdges:
    def test_max_iter_zero_returns_initial(self) -> None:
        net = _two_bus_pq(p_load_mw=50.0)
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        v0 = np.ones(net.n_bus, dtype=np.complex128)
        res = gauss_seidel(ybus, sbus, v0, ref, pv, pq, tol=1e-12, max_iter=0)
        assert not res.converged
        assert res.iterations == 0
        np.testing.assert_array_equal(res.V, v0)

    def test_negative_max_iter_raises(self) -> None:
        net = _two_bus_pq()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        v0 = np.ones(net.n_bus, dtype=np.complex128)
        with pytest.raises(ValueError, match="max_iter"):
            gauss_seidel(ybus, sbus, v0, ref, pv, pq, max_iter=-1)


class TestGaussSeidelPV:
    def test_three_bus_with_pv(self) -> None:
        """Slack — PV — PQ через две одинаковые линии. PV должен удерживать |V|."""
        n = 3
        net = NetworkPU(
            n_bus=n,
            n_branch=2,
            bus_ids=np.array([1, 2, 3], dtype=np.int64),
            bus_vn_kv=np.full(n, 110.0),
            bus_type=np.array([2, 1, 0], dtype=np.int8),
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
            # PV генерация 30 МВт, PQ нагрузка 50/20
            bus_p_injection=np.array([0.0, 0.30, -0.50]),
            bus_q_injection=np.array([0.0, 0.0, -0.20]),
            base_mva=BASE_MVA,
        )
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        # PV-шине задаём |V|=1.02
        v0 = np.array([1.0, 1.02, 1.0], dtype=np.complex128)
        res = gauss_seidel(ybus, sbus, v0, ref, pv, pq, tol=1e-8, max_iter=500)
        assert res.converged
        np.testing.assert_allclose(abs(res.V[1]), 1.02, atol=1e-8)
