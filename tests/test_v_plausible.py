"""Sanity-гейт правдоподобия |V| (v_plausible_range).

NR может численно сойтись (mismatch < tol) в нижнюю ветвь PV-кривой —
физически бессмысленный режим (|V| ~ 0.03 p.u.), который раньше репортился
``converged=True``. Гейт помечает его ``converged=False`` с
``failure_reason="implausible_voltage"``; ``v_plausible_range=None``
возвращает прежнее поведение бит-в-бит.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gridpf import PFInput, PFOptions, solve


def _two_bus_loaded(build_net: Callable[..., PFInput]) -> PFInput:
    # slack(0) — PQ(1, нагрузка 0.5+j0.2): верхнее решение ~0.984,
    # нижняя ветвь ~0.028 p.u. (достижима warm-start'ом с низким |V|).
    return build_net(
        bus_type=[2, 0],
        edges=[(0, 1, 0.01, 0.05)],
        p_inj=[0.0, -0.5],
        q_inj=[0.0, -0.2],
    )


def test_upper_branch_passes_gate(build_net: Callable[..., PFInput]) -> None:
    res = solve(_two_bus_loaded(build_net), PFOptions(method="nr"))
    assert res.converged
    assert res.failure_reason == ""
    assert res.implausible_v_nodes == 0


def test_lower_branch_flagged_implausible(build_net: Callable[..., PFInput]) -> None:
    net = _two_bus_loaded(build_net)
    init = np.array([1.0 + 0j, 0.1 + 0j])
    res = solve(net, PFOptions(method="nr", dc_fallback=False), init_v=init)
    # Численно решение есть (нижняя ветвь, mismatch < tol), но физически — нет.
    assert res.mismatch_max < 1e-8
    assert not res.converged
    assert res.failure_reason == "implausible_voltage"
    assert res.implausible_v_nodes == 1
    assert float(np.abs(res.V[1])) < 0.5


def test_gate_off_restores_previous_behavior(build_net: Callable[..., PFInput]) -> None:
    net = _two_bus_loaded(build_net)
    init = np.array([1.0 + 0j, 0.1 + 0j])
    res = solve(
        net,
        PFOptions(method="nr", dc_fallback=False, v_plausible_range=None),
        init_v=init,
    )
    assert res.converged  # прежнее поведение: численная сходимость = успех
    assert res.failure_reason == ""
    assert res.implausible_v_nodes == 0
