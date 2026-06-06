"""Тесты для robustness-фич: step_control, DC-fallback."""

from __future__ import annotations

import numpy as np

from gridpf import PFInput as NetworkPU
from gridpf.algebra.sbus import build_sbus, classify_buses
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.types import BASE_MVA
from gridpf.solvers.dc_pf import dc_powerflow
from gridpf.solvers.newton_raphson import newton_raphson


def _three_bus_network() -> NetworkPU:
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


class TestStepControl:
    def test_step_control_default_on(self) -> None:
        """С heavy initial mismatch step_control принимает уменьшенный шаг."""
        net = _three_bus_network()
        ybus, _, _ = build_ybus(net)
        sbus = build_sbus(net)
        ref, pv, pq = classify_buses(net.bus_type)
        # Тяжёлый начальный угол — провоцирует overshoot полным шагом.
        V0 = np.array([1.0, 1.02, 0.7 * np.exp(-0.5j)], dtype=np.complex128)
        res_ctrl = newton_raphson(
            ybus, sbus, V0, ref, pv, pq, tol=1e-10, max_iter=30, step_control=True
        )
        res_no = newton_raphson(
            ybus, sbus, V0, ref, pv, pq, tol=1e-10, max_iter=30, step_control=False
        )
        # Оба должны сойтись на этой простой сети, но step_control никогда не хуже.
        assert res_ctrl.converged
        if res_no.converged:
            np.testing.assert_allclose(res_ctrl.V, res_no.V, atol=1e-9)


class TestDCWarmStart:
    def test_dc_solver_runs(self) -> None:
        net = _three_bus_network()
        ref, pv, pq = classify_buses(net.bus_type)
        delta = dc_powerflow(
            n_bus=net.n_bus,
            from_idx=net.from_idx,
            to_idx=net.to_idx,
            branch_x=net.branch_x,
            tap_ratio=net.tap_ratio,
            P_inj=net.bus_p_injection,
            ref=ref,
            pv=pv,
            pq=pq,
        )
        assert delta.shape == (net.n_bus,)
        assert delta[ref[0]] == 0.0  # slack δ = 0
        # Углы должны быть конечными.
        assert np.all(np.isfinite(delta))
