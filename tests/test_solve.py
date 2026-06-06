"""Численная регрессия движка: сходимость, согласованность методов, тёплый старт,
setpoints, СХН, Q-лимиты."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from gridpf import PFInput, PFOptions, solve


def test_two_bus_converges(two_bus: PFInput) -> None:
    res = solve(two_bus, PFOptions(method="gs+nr"))
    assert res.converged
    assert res.mismatch_max < 1e-8
    vm = np.abs(res.V)
    # slack держит 1.0; нагрузочный PQ-узел просаживается, но остаётся в пределах.
    assert vm[0] == pytest.approx(1.0)
    assert 0.9 < vm[1] < 1.0


@pytest.mark.parametrize("method", ["gs+nr", "nr", "gs"])
def test_methods_agree(two_bus: PFInput, method: str) -> None:
    res = solve(two_bus, PFOptions(method=method, max_iter_gs=200, tol=1e-9))
    ref = solve(two_bus, PFOptions(method="nr", tol=1e-10))
    assert res.converged
    np.testing.assert_allclose(res.V, ref.V, atol=1e-6)


def test_warm_start_matches_and_is_cheaper(two_bus: PFInput) -> None:
    cold = solve(two_bus, PFOptions(method="nr"))
    warm = solve(two_bus, PFOptions(method="nr"), init_v=cold.V)
    np.testing.assert_allclose(warm.V, cold.V, atol=1e-8)
    # Тёплый старт из решения — не дороже холодного по NR-итерациям.
    assert warm.iterations_nr <= cold.iterations_nr


def test_init_v_none_equals_default(two_bus: PFInput) -> None:
    a = solve(two_bus, PFOptions(), init_v=None)
    b = solve(two_bus, PFOptions())
    np.testing.assert_array_equal(a.V, b.V)


def test_pv_setpoint_is_held(build_net: Callable[..., PFInput]) -> None:
    # slack(0) — PV(1, |V|=1.05, отдаёт 0.3 p.u.) — PQ(2, нагрузка 0.5+j0.2).
    net = build_net(
        bus_type=[2, 1, 0],
        edges=[(0, 1, 0.01, 0.05), (1, 2, 0.02, 0.06)],
        p_inj=[0.0, 0.3, -0.5],
        q_inj=[0.0, 0.0, -0.2],
        bus_v_set=[1.0, 1.05, np.nan],
        bus_va_set=[0.0, np.nan, np.nan],
    )
    res = solve(net, PFOptions(method="gs+nr"))
    assert res.converged
    vm = np.abs(res.V)
    assert vm[0] == pytest.approx(1.0, abs=1e-9)
    assert vm[1] == pytest.approx(1.05, abs=1e-6)  # PV |V| удержан setpoint'ом


def test_voltage_dependent_load_active(build_net: Callable[..., PFInput]) -> None:
    net = build_net(
        bus_type=[2, 0],
        edges=[(0, 1, 0.02, 0.06)],
        p_inj=[0.0, -0.5],
        q_inj=[0.0, -0.2],
        schn={"bus_p_a1": [0.0, 1.0], "bus_p_a0": [1.0, 0.0]},  # P_load ∝ |V|
    )
    assert net.has_voltage_dependent_load
    res = solve(net, PFOptions(use_load_voltage_dependency=True))
    assert res.converged
    assert res.voltage_dependent_load_active
    # При выключенной СХН расчёт тоже сходится, но флаг снят.
    res_off = solve(net, PFOptions(use_load_voltage_dependency=False))
    assert res_off.converged
    assert not res_off.voltage_dependent_load_active


def test_enforce_q_lims_runs_and_converges(build_net: Callable[..., PFInput]) -> None:
    net = build_net(
        bus_type=[2, 1, 0],
        edges=[(0, 1, 0.01, 0.05), (1, 2, 0.02, 0.06)],
        p_inj=[0.0, 0.3, -0.8],
        q_inj=[0.0, 0.0, -0.4],
        bus_v_set=[1.0, 1.05, np.nan],
        q_min=[np.nan, -0.05, np.nan],
        q_max=[np.nan, 0.05, np.nan],  # тесный Q-лимит у PV → возможен swap
    )
    res = solve(net, PFOptions(enforce_q_lims=True))
    assert res.converged
    assert res.q_lim_swaps >= 0  # не падает; swap может быть 0 или больше


def test_branch_flow_in_mva(two_bus: PFInput) -> None:
    res = solve(two_bus, PFOptions())
    # Переток от slack к нагрузке: активная часть ≈ нагрузка (0.5 p.u. × 100) + потери.
    assert res.S_from.shape == (1,)
    assert res.S_from[0].real * 1.0 > 50.0  # МВт, ≥ нагрузки


def test_no_slack_raises(build_net: Callable[..., PFInput]) -> None:
    net = build_net(
        bus_type=[0, 0],
        edges=[(0, 1, 0.02, 0.06)],
        p_inj=[0.0, -0.5],
        q_inj=[0.0, -0.2],
    )
    with pytest.raises(ValueError, match="slack"):
        solve(net, PFOptions(), validate=False)


def test_unknown_method_raises(two_bus: PFInput) -> None:
    with pytest.raises(ValueError, match="method"):
        solve(two_bus, PFOptions(method="xyz"))  # type: ignore[arg-type]
