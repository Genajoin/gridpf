"""Тесты Jacobian и Newton-Raphson итератора."""

from __future__ import annotations

import numpy as np
import pytest

from gridpf import PFInput as NetworkPU
from gridpf.algebra.jacobian import build_jacobian, dSbus_dV
from gridpf.algebra.sbus import build_sbus, classify_buses
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.types import BASE_MVA
from gridpf.solvers.gauss_seidel import gauss_seidel
from gridpf.solvers.newton_raphson import _mismatch, _residual_vector, newton_raphson


def _three_bus_network() -> NetworkPU:
    """Slack — PV — PQ через две одинаковые линии (для тестов NR/Jacobian)."""
    n = 3
    return NetworkPU(
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
        bus_p_injection=np.array([0.0, 0.30, -0.50]),
        bus_q_injection=np.array([0.0, 0.0, -0.20]),
        base_mva=BASE_MVA,
    )


class TestJacobianNumerical:
    def test_dsbus_dv_against_finite_difference(self) -> None:
        """Численная проверка dS/d|V| и dS/dδ конечными разностями."""
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        # Ненулевые углы и не-flat |V| — чтобы все производные были ненулевыми.
        Vm = np.array([1.00, 1.02, 0.96])
        Va = np.array([0.0, -0.05, -0.12])
        V = Vm * np.exp(1j * Va)
        sbus = build_sbus(net)

        dS_dVm, dS_dVa = dSbus_dV(ybus, V)
        dS_dVm_d = dS_dVm.toarray()
        dS_dVa_d = dS_dVa.toarray()

        eps = 1e-7
        n = V.size
        for k in range(n):
            # Возмущение по |V|_k
            Vm2 = Vm.copy()
            Vm2[k] += eps
            V2 = Vm2 * np.exp(1j * Va)
            dS_num = (V2 * np.conj(ybus @ V2) - V * np.conj(ybus @ V)) / eps
            np.testing.assert_allclose(dS_num, dS_dVm_d[:, k], atol=1e-6)

            # Возмущение по δ_k
            Va2 = Va.copy()
            Va2[k] += eps
            V2 = Vm * np.exp(1j * Va2)
            dS_num = (V2 * np.conj(ybus @ V2) - V * np.conj(ybus @ V)) / eps
            np.testing.assert_allclose(dS_num, dS_dVa_d[:, k], atol=1e-6)

        # подавляем неиспользование sbus, fixture создан для других тестов
        _ = sbus

    def test_active_jacobian_shape(self) -> None:
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        V = np.ones(net.n_bus, dtype=np.complex128)
        J = build_jacobian(ybus, V, pv, pq)
        # n_pv = 1, n_pq = 1 → размер (1+1) + 1 = 3 строки и 3 столбца
        assert J.shape == (pv.size + pq.size + pq.size, pv.size + pq.size + pq.size)
        _ = (sbus, ref)


class TestNewtonRaphsonConvergence:
    def test_three_bus_converges_fast(self) -> None:
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        # Flat-старт: |V|=1 на PQ; PV получает |V|=1.02
        V0 = np.array([1.0, 1.02, 1.0], dtype=np.complex128)
        res = newton_raphson(ybus, sbus, V0, ref, pv, pq, tol=1e-10, max_iter=20)
        assert res.converged
        # Newton — квадратичная сходимость; обычно хватает 3-5 итераций
        assert res.iterations <= 6
        # Slack не двигается
        np.testing.assert_allclose(res.V[0], 1.0 + 0j, atol=1e-12)
        # PV-шина удерживает заданный |V|
        np.testing.assert_allclose(abs(res.V[1]), 1.02, atol=1e-10)
        # Под нагрузкой PQ-шина просела
        assert abs(res.V[2]) < 1.0

    def test_max_iter_zero_returns_initial(self) -> None:
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        V0 = np.ones(net.n_bus, dtype=np.complex128)
        res = newton_raphson(ybus, sbus, V0, ref, pv, pq, tol=1e-12, max_iter=0)
        assert not res.converged
        assert res.iterations == 0

    def test_negative_max_iter_raises(self) -> None:
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        V0 = np.ones(net.n_bus, dtype=np.complex128)
        with pytest.raises(ValueError, match="max_iter"):
            newton_raphson(ybus, sbus, V0, ref, pv, pq, max_iter=-1)


class TestGSPlusNR:
    def test_gs_warm_start_then_newton(self) -> None:
        """Связка GS + NR на 3-узловой схеме — то же решение, что и чистый NR."""
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        V0 = np.array([1.0, 1.02, 1.0], dtype=np.complex128)

        gs_res = gauss_seidel(ybus, sbus, V0, ref, pv, pq, tol=1e-2, max_iter=10)
        nr_res = newton_raphson(ybus, sbus, gs_res.V, ref, pv, pq, tol=1e-10, max_iter=20)
        nr_pure = newton_raphson(ybus, sbus, V0, ref, pv, pq, tol=1e-10, max_iter=20)

        assert nr_res.converged and nr_pure.converged
        np.testing.assert_allclose(nr_res.V, nr_pure.V, atol=1e-9)


def test_residual_vector_helper() -> None:
    pv = np.array([1])
    pq = np.array([2])
    mis = np.array([0.0 + 0j, 1.0 + 2j, 3.0 + 4j])
    f = _residual_vector(mis, pv, pq)
    np.testing.assert_array_equal(f, [1.0, 3.0, 4.0])


def test_mismatch_helper() -> None:
    from scipy.sparse import csr_matrix as sparse

    Ybus = sparse(np.eye(2, dtype=np.complex128) * 2)
    V = np.array([1.0, 1.0], dtype=np.complex128)
    Sbus = np.array([1.0 + 0j, 0.0])
    # mis = V*conj(Y*V) - S = 1*2 - 1 = 1, 1*2 - 0 = 2
    np.testing.assert_allclose(_mismatch(Ybus, V, Sbus), [1.0, 2.0])
