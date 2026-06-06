"""Тесты Sbus: построение Sbus и классификация шин."""

from __future__ import annotations

import numpy as np
import pytest

from gridpf import PFInput as NetworkPU
from gridpf.algebra.sbus import build_sbus, classify_buses
from gridpf.contract.types import BASE_MVA, PQ, PV, SLACK


def _make_network(p_inj: list[float], q_inj: list[float], bus_type: list[int]) -> NetworkPU:
    n = len(p_inj)
    return NetworkPU(
        n_bus=n,
        n_branch=0,
        bus_ids=np.arange(1, n + 1, dtype=np.int64),
        bus_vn_kv=np.full(n, 110.0),
        bus_type=np.array(bus_type, dtype=np.int8),
        slack_idx=int(np.where(np.array(bus_type) == SLACK)[0][0]),
        branch_ids=np.empty(0, dtype=np.int64),
        from_idx=np.empty(0, dtype=np.int64),
        to_idx=np.empty(0, dtype=np.int64),
        branch_r=np.empty(0),
        branch_x=np.empty(0),
        branch_g=np.empty(0),
        branch_b=np.empty(0),
        branch_g_from=np.empty(0),
        branch_b_from=np.empty(0),
        branch_g_to=np.empty(0),
        branch_b_to=np.empty(0),
        tap_ratio=np.empty(0),
        phase_shift=np.empty(0),
        bus_g_shunt=np.zeros(n),
        bus_b_shunt=np.zeros(n),
        bus_p_injection=np.array(p_inj, dtype=np.float64),
        bus_q_injection=np.array(q_inj, dtype=np.float64),
        base_mva=BASE_MVA,
    )


class TestBuildSbus:
    def test_complex_combine(self) -> None:
        net = _make_network(
            p_inj=[0.5, -0.2, 0.0],
            q_inj=[0.1, -0.05, 0.3],
            bus_type=[SLACK, PQ, PV],
        )
        sbus = build_sbus(net)
        assert sbus.dtype == np.complex128
        np.testing.assert_allclose(sbus.real, [0.5, -0.2, 0.0])
        np.testing.assert_allclose(sbus.imag, [0.1, -0.05, 0.3])

    def test_zero_injections(self) -> None:
        net = _make_network([0, 0], [0, 0], [SLACK, PQ])
        sbus = build_sbus(net)
        assert np.all(sbus == 0)


class TestClassifyBuses:
    def test_three_types(self) -> None:
        bus_type = np.array([SLACK, PQ, PV, PQ, PV], dtype=np.int8)
        ref, pv, pq = classify_buses(bus_type)
        np.testing.assert_array_equal(ref, [0])
        np.testing.assert_array_equal(pv, [2, 4])
        np.testing.assert_array_equal(pq, [1, 3])

    def test_no_slack_raises(self) -> None:
        with pytest.raises(ValueError, match="нет slack-шины"):
            classify_buses(np.array([PQ, PV, PQ], dtype=np.int8))

    def test_only_slack(self) -> None:
        ref, pv, pq = classify_buses(np.array([SLACK], dtype=np.int8))
        assert ref.tolist() == [0]
        assert pv.size == 0
        assert pq.size == 0
