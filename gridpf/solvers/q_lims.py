"""Опциональное переключение PV ↔ PQ по ограничениям ``Q_min``/``Q_max``.

Используется во внешнем цикле движка при ``enforce_q_lims=True``:

1. После сошедшегося NR на текущих ``pv``/``pq`` массивах считаем фактический
   ``Q_calc`` на каждой PV-шине (для PQ — Q зафиксирован).
2. На PV-шине с ``Q_calc > Q_max`` фиксируем ``Q = Q_max`` и переводим узел в
   PQ (его ``|V|`` теперь свободна). Аналогично для ``Q_calc < Q_min``.
3. На «изначально PV, но переведённой в PQ» шине проверяем обратное:
   если она удерживала ``Q_max`` и теперь ``|V|_calc > V_set`` (значит,
   нужна была бы меньшая Q) — возвращаем в PV. Симметрично для ``Q_min``.

Лимит итераций внешнего цикла защищает от осцилляций (узлы могут
циклически переключаться при граничных условиях).

Конвенция: ``q_min``, ``q_max`` в **p.u.**; ``NaN`` означает «лимит не
задан». Если для PV-шины оба лимита NaN — она не участвует в проверке.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


@dataclass
class QLimAction:
    """Лог одной операции переключения."""

    bus_idx: int
    direction: str  # "pv->pq_qmax" / "pv->pq_qmin" / "pq->pv_qmax" / "pq->pv_qmin"
    q_value: float  # на PV→PQ — какое Q зафиксировали; на обратном — q_calc


@dataclass
class QLimResult:
    """Результат одной итерации проверки Q-лимитов."""

    pv: np.ndarray
    pq: np.ndarray
    Sbus: np.ndarray
    v_set: np.ndarray  # заданные |V| для PV (постоянны; для не-PV игнорируется)
    locked_lim: np.ndarray  # для каждого узла: NaN, или значение Q_max/Q_min, на котором закреплён
    actions: list[QLimAction]
    changed: bool
    bus_q_gen_new: np.ndarray | None = None
    """Обновлённые значения ``bus_q_gen`` после swap'а — нужно при работе с СХН.
    На swap'нутых узлах содержит зафиксированный ``Q_lim`` (предел генератора).
    ``None`` — если ``network_pu`` не передан (СХН не активна, swap пишет
    ``Q_inj`` напрямую через ``Sbus``)."""


def enforce_q_limits(
    Ybus: csr_matrix,
    V: np.ndarray,
    Sbus: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    v_set: np.ndarray,
    locked_lim: np.ndarray,
    *,
    pv_original: np.ndarray,
    allow_pq_to_pv: bool = False,
    top_k: int | None = None,
    network_pu: PFInput | None = None,
) -> QLimResult:
    """Применить Q-лимиты после очередного NR-прогона.

    Args:
        Ybus: ``(n, n)`` CSR.
        V: ``(n,)`` complex — текущее напряжение (после NR).
        Sbus: ``(n,)`` complex — текущие инъекции (Q PV-шин подменяется при
            переводе в PQ; не модифицируется).
        pv: индексы PV-шин на текущей итерации.
        pq: индексы PQ-шин на текущей итерации.
        q_min, q_max: ``(n,)`` — лимиты Q **генератора** в p.u. (NaN = не задан).
        v_set: ``(n,)`` — заданные |V| для исходных PV (NaN если не задано).
        locked_lim: ``(n,)`` — на каком Q-лимите закреплён узел; NaN если не закреплён.
        pv_original: индексы шин, которые в исходной модели были PV
            (нужны для обратного перевода PQ → PV).
        network_pu: при наличии включает корректное взаимодействие с СХН.
            Q-лимит трактуется как лимит **генератора** (``Q_gen``), а не
            суммарной инъекции; ``Q_load(|V|)`` вычитается из ``Q_calc``.
            При swap'е PV→PQ возвращается обновлённый ``bus_q_gen_new``,
            который вызывающий код подаёт в ``compute_sbus`` для пересчёта
            ``Sbus`` с учётом переменной нагрузки. Без СХН swap пишет
            ``Q_inj`` напрямую в ``Sbus`` (legacy-поведение).

    Returns:
        :class:`QLimResult` с обновлёнными ``pv``/``pq``/``Sbus``,
        флагом ``changed`` и списком ``actions``. При ``network_pu`` задан —
        также ``bus_q_gen_new``.
    """
    pv_set = set(pv.tolist())
    pq_set = set(pq.tolist())
    Sbus_new = Sbus.copy()
    locked_new = locked_lim.copy()
    actions: list[QLimAction] = []

    # При активной СХН Q-лимит относится к генератору, поэтому из Q_calc
    # (сетевая Q-инъекция) вычитаем текущую Q_load(|V|), чтобы получить Q_gen.
    use_load = network_pu is not None and network_pu.has_voltage_dependent_load
    if use_load:
        assert network_pu is not None
        Vm_now = np.abs(V)
        b0, b1, b2 = network_pu.bus_q_b0, network_pu.bus_q_b1, network_pu.bus_q_b2
        q_load_arr = network_pu.bus_q_load
        assert b0 is not None and b1 is not None and b2 is not None and q_load_arr is not None
        q_load_at_v = q_load_arr * (b0 + b1 * Vm_now + b2 * Vm_now * Vm_now)
        bus_q_gen_new: np.ndarray | None = (
            network_pu.bus_q_gen.copy() if network_pu.bus_q_gen is not None else None
        )
    else:
        q_load_at_v = None
        bus_q_gen_new = None

    # ---- 1. PV → PQ: проверяем Q_gen на текущих PV-шинах
    if pv.size > 0:
        I_bus = Ybus @ V
        S_calc = V * np.conj(I_bus)
        # Сначала собираем список нарушителей с величиной превышения, потом
        # переключаем либо всех (top_k=None), либо top-k самых тяжёлых.
        violators: list[tuple[float, int, float, str]] = []
        for k in pv.tolist():
            qk_calc = float(S_calc[k].imag)
            # Q_gen = Q_inj + Q_load(|V|); без СХН Q_load=0 → Q_gen = Q_inj.
            qk_gen = qk_calc + (float(q_load_at_v[k]) if q_load_at_v is not None else 0.0)
            qmax_k = float(q_max[k])
            qmin_k = float(q_min[k])
            if not np.isnan(qmax_k) and qk_gen > qmax_k:
                violators.append((qk_gen - qmax_k, k, qmax_k, "qmax"))
            elif not np.isnan(qmin_k) and qk_gen < qmin_k:
                violators.append((qmin_k - qk_gen, k, qmin_k, "qmin"))

        if top_k is not None and top_k > 0 and len(violators) > top_k:
            violators.sort(reverse=True)  # по убыванию excess
            violators = violators[:top_k]

        for _excess, k, qlim, direction in violators:
            # Без СХН: Sbus.imag = qlim напрямую (Q_inj = Q_gen - 0).
            # С СХН: фиксируем Q_gen = qlim; Sbus.imag = qlim - Q_load(|V|).
            #   Дальше caller пересоберёт Sbus через compute_sbus уже на каждом
            #   шаге NR, поэтому конкретное значение в Sbus_new — стартовое.
            if q_load_at_v is not None:
                Sbus_new[k] = Sbus_new[k].real + 1j * (qlim - float(q_load_at_v[k]))
                if bus_q_gen_new is not None:
                    bus_q_gen_new[k] = qlim
            else:
                Sbus_new[k] = Sbus_new[k].real + 1j * qlim
            locked_new[k] = qlim
            pv_set.discard(k)
            pq_set.add(k)
            actions.append(
                QLimAction(
                    bus_idx=k,
                    direction="pv->pq_qmax" if direction == "qmax" else "pv->pq_qmin",
                    q_value=qlim,
                )
            )

    # ---- 2. PQ → PV: для шин, ранее закреплённых на лимите
    # По умолчанию обратное переключение запрещено (MATPOWER-default) — это
    # резко стабилизирует outer-loop при тесных Q-лимитах. Включается только
    # с allow_pq_to_pv=True.
    if not allow_pq_to_pv:
        pv_new = np.array(sorted(pv_set), dtype=np.int64)
        pq_new = np.array(sorted(pq_set), dtype=np.int64)
        return QLimResult(
            pv=pv_new,
            pq=pq_new,
            Sbus=Sbus_new,
            v_set=v_set,
            locked_lim=locked_new,
            actions=actions,
            changed=len(actions) > 0,
            bus_q_gen_new=bus_q_gen_new,
        )

    pv_orig_set = set(pv_original.tolist())
    for k in pq.tolist():
        if k not in pv_orig_set or np.isnan(locked_new[k]):
            continue
        vk = float(np.abs(V[k]))
        vk_set = float(v_set[k])
        if np.isnan(vk_set):
            continue
        # Закреплён на Q_max → если |V| вырос выше V_set, узлу нужна была бы
        # меньшая Q; возвращаем в PV (с заданным |V|).
        qlocked = float(locked_new[k])
        qmax = q_max[k]
        qmin = q_min[k]
        # Dead-band 0.01 p.u. (≈1 % от V_set) — защищает от осцилляций
        # PV ↔ PQ при граничных условиях. MATPOWER-конвенция.
        v_deadband = 1e-2
        if not np.isnan(qmax) and qlocked == qmax and vk > vk_set + v_deadband:
            pq_set.discard(k)
            pv_set.add(k)
            locked_new[k] = np.nan
            actions.append(QLimAction(bus_idx=k, direction="pq->pv_qmax", q_value=vk))
        elif not np.isnan(qmin) and qlocked == qmin and vk < vk_set - v_deadband:
            pq_set.discard(k)
            pv_set.add(k)
            locked_new[k] = np.nan
            actions.append(QLimAction(bus_idx=k, direction="pq->pv_qmin", q_value=vk))

    pv_new = np.array(sorted(pv_set), dtype=np.int64)
    pq_new = np.array(sorted(pq_set), dtype=np.int64)
    return QLimResult(
        pv=pv_new,
        pq=pq_new,
        Sbus=Sbus_new,
        v_set=v_set,
        locked_lim=locked_new,
        actions=actions,
        changed=len(actions) > 0,
        bus_q_gen_new=bus_q_gen_new,
    )
