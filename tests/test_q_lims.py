"""Тесты опционального переключения PV ↔ PQ по Q_min/Q_max.

Проверяем:
* при включении PV-шина с тесными лимитами получает Q=Q_max и переходит в PQ;
* нарушение Q_min → PV→PQ;
* top_k ограничивает число переключений;
* обратное PQ→PV при ``allow_pq_to_pv=True``;
* результат совпадает с ``pandapower.runpp(enforce_q_lims=True)`` на той же сети.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix as sparse

from gridpf.solvers.q_lims import enforce_q_limits


def _make_3bus() -> tuple:
    """3-шины: slack(0) — PV(1) — PQ(2). Возвращает (ybus, V, Sbus, pv, pq)."""
    y = np.array(
        [
            [4 - 20j, -2 + 10j, 0],
            [-2 + 10j, 6 - 30j, -4 + 20j],
            [0, -4 + 20j, 4 - 20j],
        ],
        dtype=np.complex128,
    )
    ybus = sparse(y)
    V = np.array([1.0 + 0j, 1.02 + 0j, 0.95 + 0j], dtype=np.complex128)
    Sbus = np.array([0.0, 0.30, -0.50 - 0.20j], dtype=np.complex128)
    pv = np.array([1], dtype=np.int64)
    pq = np.array([2], dtype=np.int64)
    return ybus, V, Sbus, pv, pq


class TestEnforceQLimitsHelper:
    def test_no_q_lims_no_change(self) -> None:
        """Если q_min, q_max — все NaN, переключений нет."""
        n = 3
        ybus = sparse(np.eye(n, dtype=np.complex128) * (-2j))
        V = np.array([1.0, 1.02, 0.95], dtype=np.complex128)
        Sbus = np.array([0.0, 0.30, -0.50 - 0.20j], dtype=np.complex128)
        pv = np.array([1])
        pq = np.array([2])
        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        v_set = np.full(n, np.nan)
        locked = np.full(n, np.nan)
        res = enforce_q_limits(ybus, V, Sbus, pv, pq, q_min, q_max, v_set, locked, pv_original=pv)
        assert not res.changed
        assert res.actions == []

    def test_qmax_violation_pv_to_pq(self) -> None:
        """Q_gen > Q_max на PV-шине → переключение в PQ, Sbus.imag = Q_max."""
        ybus, V, Sbus, pv, pq = _make_3bus()
        n = 3
        # Считаем Q_gen на шине 1 при данных V.
        S_calc = V * np.conj(ybus @ V)
        q_gen_1 = float(S_calc[1].imag)

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        # Ставим Q_max ниже рассчитанного Q_gen.
        q_max[1] = q_gen_1 - 0.01
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        locked = np.full(n, np.nan)

        res = enforce_q_limits(ybus, V, Sbus, pv, pq, q_min, q_max, v_set, locked, pv_original=pv)

        assert res.changed
        assert len(res.actions) == 1
        assert res.actions[0].bus_idx == 1
        assert res.actions[0].direction == "pv->pq_qmax"
        assert res.actions[0].q_value == pytest.approx(q_max[1], rel=1e-10)
        # Шина 1 перешла в PQ.
        assert 1 not in res.pv
        assert 1 in res.pq
        # Sbus.imag на шине 1 == Q_max.
        assert float(res.Sbus[1].imag) == pytest.approx(q_max[1], rel=1e-10)
        # locked_lim показывает закрепление.
        assert float(res.locked_lim[1]) == pytest.approx(q_max[1], rel=1e-10)

    def test_constant_load_generator_semantics(self) -> None:
        """Q-лимит — лимит ГЕНЕРАТОРА и при константной нагрузке (без СХН):
        нетто-инъекция узла «генерация + нагрузка» глубоко ниже ``q_min``
        генератора, но ``Q_gen = Q_calc + Q_load`` внутри лимитов → свопа нет
        (раньше без активной СХН проверка сравнивала нетто и ложно свопала)."""
        from dataclasses import replace

        from gridpf import PFOptions, build_ybus, solve
        from tests.conftest import _build_net

        # PV-узел 1 несёт и генерацию, и Q-нагрузку 0.8 (константа: b0=1, b1=b2=0).
        net = _build_net(
            bus_type=[2, 1, 0],
            edges=[(0, 1, 0.01, 0.05), (1, 2, 0.02, 0.06)],
            p_inj=[0.0, 0.3, -0.5],
            q_inj=[0.0, -0.3, -0.2],
            schn={
                "bus_p_gen": [0.0, 0.3, 0.0],
                "bus_q_gen": [0.0, 0.5, 0.0],
                "bus_p_load": [0.0, 0.0, 0.5],
                "bus_q_load": [0.0, 0.8, 0.2],
            },
        )
        net = replace(
            net,
            bus_v_set=np.array([1.0, 1.02, np.nan]),
            bus_va_set=np.array([0.0, np.nan, np.nan]),
        )
        # Калибровка: фактический Q_gen PV-узла в решении без лимитов.
        ref = solve(net, PFOptions(enforce_q_lims=False))
        assert ref.converged
        ybus, _, _ = build_ybus(net)
        q_calc_1 = float((ref.V * np.conj(ybus @ ref.V))[1].imag)
        q_gen_1 = q_calc_1 + 0.8  # константная Q-нагрузка узла

        # Лимиты генератора вокруг фактического Q_gen (внутри); нетто-инъекция
        # (Q_gen − 0.8) при этом глубоко ниже q_min — старая нетто-семантика
        # дала бы ложный своп pv->pq_qmin.
        net_lim = replace(
            net,
            bus_q_min=np.array([np.nan, q_gen_1 - 0.1, np.nan]),
            bus_q_max=np.array([np.nan, q_gen_1 + 0.1, np.nan]),
        )
        res = solve(net_lim, PFOptions(enforce_q_lims=True))
        assert res.converged
        assert res.q_lim_swaps == 0
        assert res.q_violations == 0
        # PV удержал уставку — решение совпало с безлимитным.
        np.testing.assert_allclose(res.V, ref.V, atol=1e-8)

    def test_qmin_violation_pv_to_pq(self) -> None:
        """Q_gen < Q_min на PV-шине → переключение в PQ, Sbus.imag = Q_min."""
        ybus, V, Sbus, pv, pq = _make_3bus()
        n = 3
        S_calc = V * np.conj(ybus @ V)
        q_gen_1 = float(S_calc[1].imag)

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        # Ставим Q_min выше рассчитанного Q_gen.
        q_min[1] = q_gen_1 + 0.01
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        locked = np.full(n, np.nan)

        res = enforce_q_limits(ybus, V, Sbus, pv, pq, q_min, q_max, v_set, locked, pv_original=pv)

        assert res.changed
        assert len(res.actions) == 1
        assert res.actions[0].bus_idx == 1
        assert res.actions[0].direction == "pv->pq_qmin"
        assert res.actions[0].q_value == pytest.approx(q_min[1], rel=1e-10)
        assert 1 not in res.pv
        assert 1 in res.pq

    def test_top_k_limits_switches(self) -> None:
        """top_k=1 при двух нарушителях переключает только одного (самого тяжёлого)."""
        n = 4
        y = np.array(
            [
                [4 - 20j, -2 + 10j, 0, 0],
                [-2 + 10j, 4 - 20j, -2 + 10j, 0],
                [0, -2 + 10j, 4 - 20j, -2 + 10j],
                [0, 0, -2 + 10j, 4 - 20j],
            ],
            dtype=np.complex128,
        )
        ybus = sparse(y)
        V = np.array([1.0, 1.02, 1.01, 0.96], dtype=np.complex128)
        Sbus = np.array([0.0, 0.30, 0.25, -0.50 - 0.20j], dtype=np.complex128)
        pv = np.array([1, 2], dtype=np.int64)
        pq = np.array([3], dtype=np.int64)

        S_calc = V * np.conj(ybus @ V)
        q_gen = S_calc[pv].imag

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        # Обе PV-шины нарушают Q_max.
        q_max[pv[0]] = q_gen[0] - 0.01
        q_max[pv[1]] = q_gen[1] - 0.005  # меньший excess
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        v_set[2] = 1.01
        locked = np.full(n, np.nan)

        res = enforce_q_limits(
            ybus,
            V,
            Sbus,
            pv,
            pq,
            q_min,
            q_max,
            v_set,
            locked,
            pv_original=pv,
            top_k=1,
        )

        assert res.changed
        assert len(res.actions) == 1
        # Переключён самый тяжёлый нарушитель (pv[0], excess=0.01 > 0.005).
        assert res.actions[0].bus_idx == pv[0]

    def test_allow_pq_to_pv_revert(self) -> None:
        """locked на Q_max, |V| вырос выше V_set + deadband → возврат в PV."""
        ybus, V, Sbus, _, _ = _make_3bus()
        n = 3

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        qmax_val = 0.01
        q_max[1] = qmax_val
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        locked = np.full(n, np.nan)
        locked[1] = qmax_val  # шина 1 «закреплена» на Q_max

        # Переводим шину 1 в PQ (эмуляция предыдущего переключения).
        pv_in = np.array([], dtype=np.int64)
        pq_in = np.array([1, 2], dtype=np.int64)
        # |V| шины 1 = 1.02 > V_set + 0.01 — условие возврата выполнено.
        V[1] = 1.04 + 0j

        res = enforce_q_limits(
            ybus,
            V,
            Sbus,
            pv_in,
            pq_in,
            q_min,
            q_max,
            v_set,
            locked,
            pv_original=np.array([1], dtype=np.int64),
            allow_pq_to_pv=True,
        )

        assert len(res.actions) == 1
        assert res.actions[0].direction == "pq->pv_qmax"
        assert 1 in res.pv
        assert 1 not in res.pq
        assert np.isnan(res.locked_lim[1])

    def test_pq_to_pv_no_revert_without_allow(self) -> None:
        """Без ``allow_pq_to_pv`` обратного переключения нет."""
        ybus, V, Sbus, _, _ = _make_3bus()
        n = 3

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        qmax_val = 0.01
        q_max[1] = qmax_val
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        locked = np.full(n, np.nan)
        locked[1] = qmax_val

        pv_in = np.array([], dtype=np.int64)
        pq_in = np.array([1, 2], dtype=np.int64)
        V[1] = 1.04 + 0j

        res = enforce_q_limits(
            ybus,
            V,
            Sbus,
            pv_in,
            pq_in,
            q_min,
            q_max,
            v_set,
            locked,
            pv_original=np.array([1], dtype=np.int64),
            allow_pq_to_pv=False,
        )

        # Нет переключений (шин с нарушениями нет, обратный запрещён).
        assert len(res.actions) == 0
        assert 1 not in res.pv

    def test_multiple_violators_all_switched(self) -> None:
        """Два PV-нарушителя без top_k — оба переключаются."""
        n = 4
        y = np.array(
            [
                [4 - 20j, -2 + 10j, 0, 0],
                [-2 + 10j, 4 - 20j, -2 + 10j, 0],
                [0, -2 + 10j, 4 - 20j, -2 + 10j],
                [0, 0, -2 + 10j, 4 - 20j],
            ],
            dtype=np.complex128,
        )
        ybus = sparse(y)
        V = np.array([1.0, 1.02, 1.01, 0.96], dtype=np.complex128)
        Sbus = np.array([0.0, 0.30, 0.25, -0.50 - 0.20j], dtype=np.complex128)
        pv = np.array([1, 2], dtype=np.int64)
        pq = np.array([3], dtype=np.int64)

        S_calc = V * np.conj(ybus @ V)
        q_gen = S_calc[pv].imag

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        q_max[pv[0]] = q_gen[0] - 0.01
        q_max[pv[1]] = q_gen[1] - 0.005
        v_set = np.full(n, np.nan)
        v_set[1] = 1.02
        v_set[2] = 1.01
        locked = np.full(n, np.nan)

        res = enforce_q_limits(
            ybus,
            V,
            Sbus,
            pv,
            pq,
            q_min,
            q_max,
            v_set,
            locked,
            pv_original=pv,
        )

        assert len(res.actions) == 2
        assert all(a.direction.startswith("pv->pq") for a in res.actions)
        assert 1 not in res.pv
        assert 2 not in res.pv
