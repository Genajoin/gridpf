"""Общие фикстуры тестов gridpf.

Помощник :func:`build_net` собирает :class:`gridpf.PFInput` из компактной
спецификации (типы шин, рёбра ``(from, to, r, x)``, инъекции p.u.) с разумными
дефолтами — чтобы тесты движка не повторяли весь конструктор контракта.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from gridpf import PFInput


def _build_net(
    bus_type: Sequence[int],
    edges: Sequence[tuple[int, int, float, float]],
    p_inj: Sequence[float],
    q_inj: Sequence[float],
    *,
    vn_kv: float = 110.0,
    slack_idx: int | None = None,
    bus_v_set: Sequence[float] | None = None,
    bus_va_set: Sequence[float] | None = None,
    q_min: Sequence[float] | None = None,
    q_max: Sequence[float] | None = None,
    schn: dict[str, Sequence[float]] | None = None,
) -> PFInput:
    """Собрать ``PFInput`` из компактной спецификации.

    Args:
        bus_type: коды типов шин (0=PQ, 1=PV, 2=SLACK).
        edges: список ``(from_pos, to_pos, r_pu, x_pu)``.
        p_inj, q_inj: инъекции на шину (p.u., gen − load).
        schn: опц. ``{"bus_p_a1": [...], ...}`` для полиномиальной СХН; требует
            ``bus_p_load`` / ``bus_q_load`` (заполняются из −p_inj/−q_inj, если
            не заданы).
    """
    n_bus = len(bus_type)
    n_branch = len(edges)
    bt = np.asarray(bus_type, dtype=np.int8)
    if slack_idx is None:
        slack_pos = np.where(bt == 2)[0]
        slack_idx = int(slack_pos[0]) if slack_pos.size else 0

    from_idx = np.array([e[0] for e in edges], dtype=np.int64)
    to_idx = np.array([e[1] for e in edges], dtype=np.int64)
    branch_r = np.array([e[2] for e in edges], dtype=np.float64)
    branch_x = np.array([e[3] for e in edges], dtype=np.float64)

    def _opt(arr: Sequence[float] | None) -> np.ndarray | None:
        return None if arr is None else np.asarray(arr, dtype=np.float64)

    schn_arrays: dict[str, np.ndarray] = {}
    if schn is not None:
        p_load = np.asarray(schn.get("bus_p_load", [-v for v in p_inj]), dtype=np.float64)
        q_load = np.asarray(schn.get("bus_q_load", [-v for v in q_inj]), dtype=np.float64)
        p_gen = np.asarray(schn.get("bus_p_gen", np.zeros(n_bus)), dtype=np.float64)
        q_gen = np.asarray(schn.get("bus_q_gen", np.zeros(n_bus)), dtype=np.float64)
        schn_arrays = {
            "bus_p_gen": p_gen,
            "bus_q_gen": q_gen,
            "bus_p_load": p_load,
            "bus_q_load": q_load,
            "bus_p_a0": np.asarray(schn.get("bus_p_a0", np.ones(n_bus)), dtype=np.float64),
            "bus_p_a1": np.asarray(schn.get("bus_p_a1", np.zeros(n_bus)), dtype=np.float64),
            "bus_p_a2": np.asarray(schn.get("bus_p_a2", np.zeros(n_bus)), dtype=np.float64),
            "bus_q_b0": np.asarray(schn.get("bus_q_b0", np.ones(n_bus)), dtype=np.float64),
            "bus_q_b1": np.asarray(schn.get("bus_q_b1", np.zeros(n_bus)), dtype=np.float64),
            "bus_q_b2": np.asarray(schn.get("bus_q_b2", np.zeros(n_bus)), dtype=np.float64),
        }

    return PFInput(
        n_bus=n_bus,
        n_branch=n_branch,
        bus_ids=np.arange(10, 10 + n_bus, dtype=np.int64),
        bus_vn_kv=np.full(n_bus, vn_kv, dtype=np.float64),
        bus_type=bt,
        slack_idx=slack_idx,
        branch_ids=np.arange(100, 100 + n_branch, dtype=np.int64),
        from_idx=from_idx,
        to_idx=to_idx,
        branch_r=branch_r,
        branch_x=branch_x,
        branch_g=np.zeros(n_branch),
        branch_b=np.zeros(n_branch),
        branch_g_from=np.zeros(n_branch),
        branch_b_from=np.zeros(n_branch),
        branch_g_to=np.zeros(n_branch),
        branch_b_to=np.zeros(n_branch),
        tap_ratio=np.ones(n_branch),
        phase_shift=np.zeros(n_branch),
        bus_g_shunt=np.zeros(n_bus),
        bus_b_shunt=np.zeros(n_bus),
        bus_p_injection=np.asarray(p_inj, dtype=np.float64),
        bus_q_injection=np.asarray(q_inj, dtype=np.float64),
        bus_q_min=_opt(q_min),
        bus_q_max=_opt(q_max),
        bus_v_set=_opt(bus_v_set),
        bus_va_set=_opt(bus_va_set),
        **schn_arrays,
    )


@pytest.fixture
def build_net() -> Callable[..., PFInput]:
    """Фабрика ``PFInput`` (см. :func:`_build_net`)."""
    return _build_net


@pytest.fixture
def two_bus(build_net: Callable[..., PFInput]) -> PFInput:
    """slack(0) — PQ(1) с нагрузкой 0.5 + j0.2 p.u. по ветви 0.02 + j0.06."""
    return build_net(
        bus_type=[2, 0],
        edges=[(0, 1, 0.02, 0.06)],
        p_inj=[0.0, -0.5],
        q_inj=[0.0, -0.2],
    )
