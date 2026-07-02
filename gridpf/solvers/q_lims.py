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

from gridpf.algebra.sbus import q_load_at


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


# Dead-band on the PQ→PV reverse swap: ~1% of V_set. Guards against PV↔PQ
# oscillation at boundary conditions (MATPOWER convention).
V_SET_DEADBAND = 1e-2


def q_limit_violations(
    q_gen: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    buses: np.ndarray,
    *,
    tol: float = 0.0,
) -> list[tuple[float, int, float, str]]:
    """Find buses whose ``q_gen`` leaves the deadband ``[q_min − tol, q_max + tol]``.

    Single home of the violation predicate — both the PV→PQ swap loop and the
    final ``q_violations`` report in the engine must agree on it.

    Args:
        q_gen: ``(n,)`` — generator reactive output per bus (``Q_inj + Q_load``).
        q_min, q_max: ``(n,)`` — generator limits in p.u.; NaN = "not set",
            such a limit never fires.
        buses: candidate bus indices (e.g. current PV set).
        tol: deadband width in p.u.

    Returns:
        ``(excess, bus, limit, kind)`` tuples in ``buses`` order, ``kind`` is
        ``"qmax"`` / ``"qmin"``. A qmax violation shadows a simultaneous qmin
        one (impossible for sane limits, mirrors the historical elif).
    """
    b = np.asarray(buses, dtype=np.int64)
    if b.size == 0:
        return []
    qg = q_gen[b]
    qmx = q_max[b]
    qmn = q_min[b]
    over = ~np.isnan(qmx) & (qg > qmx + tol)
    under = ~np.isnan(qmn) & (qg < qmn - tol) & ~over
    violators: list[tuple[float, int, float, str]] = []
    for i in np.nonzero(over | under)[0].tolist():
        if over[i]:
            violators.append((float(qg[i] - qmx[i]), int(b[i]), float(qmx[i]), "qmax"))
        else:
            violators.append((float(qmn[i] - qg[i]), int(b[i]), float(qmn[i]), "qmin"))
    return violators


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
    locked_lim: np.ndarray  # для каждого узла: NaN, или значение Q_max/Q_min, на котором закреплён
    actions: list[QLimAction]
    changed: bool
    bus_q_gen_new: np.ndarray | None = None
    """Обновлённые значения ``bus_q_gen`` после swap'а — нужно при работе с СХН.
    На swap'нутых узлах содержит зафиксированный ``Q_lim`` (предел генератора).
    ``None`` — если ``net`` не передан (СХН не активна, swap пишет
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
    q_lim_tol: float = 0.0,
    net: PFInput | None = None,
    voltage_dependent_load: bool = True,
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
        q_lim_tol: deadband на нарушение лимита (p.u.): своп PV → PQ только
            при ``Q_gen > Q_max + tol`` / ``Q_gen < Q_min − tol``. ``0.0``
            (default) — строгая проверка, прежнее поведение бит-в-бит.
            При свопе фиксируется сам лимит (не ``лимит ± tol``).
        net: при наличии (и заполненном ``bus_q_load``) включает
            ГЕНЕРАТОРНУЮ семантику: Q-лимит трактуется как лимит
            **генератора** (``Q_gen``), а не суммарной инъекции;
            ``Q_load(|V|)`` вычитается из ``Q_calc``. Семантика НЕ зависит
            от нетривиальности СХН — при тривиальных/отсутствующих
            коэффициентах нагрузка константна (``Q_load = bus_q_load``),
            но вычитать её для сравнения с лимитом генератора всё равно
            обязательно (иначе узел «генерация + нагрузка» сравнивает
            НЕТТО-инъекцию с лимитами генератора — ложные свопы).
            При swap'е PV→PQ возвращается обновлённый ``bus_q_gen_new``,
            который вызывающий код при активной СХН подаёт в
            ``compute_sbus`` для пересчёта ``Sbus``. Без ``net``
            swap пишет ``Q_inj`` напрямую в ``Sbus`` (legacy-поведение,
            Q-лимит = лимит нетто-инъекции).
        voltage_dependent_load: активна ли СХН в текущем расчёте. ``True`` —
            ``Q_load(|V|)`` по полиному; ``False`` — нагрузка константна
            (``Q_load = bus_q_load``, как в ``build_sbus``).

    Returns:
        :class:`QLimResult` с обновлёнными ``pv``/``pq``/``Sbus``,
        флагом ``changed`` и списком ``actions``. При ``net`` задан —
        также ``bus_q_gen_new``.
    """
    pv_set = set(pv.tolist())
    pq_set = set(pq.tolist())
    Sbus_new = Sbus.copy()
    locked_new = locked_lim.copy()
    actions: list[QLimAction] = []

    # Q-лимит относится к ГЕНЕРАТОРУ, поэтому из Q_calc (сетевая Q-инъекция)
    # вычитаем текущую Q_load, чтобы получить Q_gen. С активной СХН нагрузка
    # зависит от |V| (полином); без неё — константа bus_q_load (тот же смысл).
    # Гейт только на наличие данных, НЕ на нетривиальность СХН: раньше сеть
    # с константными нагрузками сравнивала нетто-инъекцию с лимитами
    # генератора и массово ложно свопала узлы «генерация + нагрузка».
    use_load = net is not None and net.bus_q_load is not None
    if use_load:
        assert net is not None
        q_load_at_v = q_load_at(net, V, voltage_dependent=voltage_dependent_load)
        bus_q_gen_new: np.ndarray | None = (
            net.bus_q_gen.copy() if net.bus_q_gen is not None else None
        )
    else:
        q_load_at_v = None
        bus_q_gen_new = None

    # ---- 1. PV → PQ: проверяем Q_gen на текущих PV-шинах
    if pv.size > 0:
        I_bus = Ybus @ V
        S_calc = V * np.conj(I_bus)
        # Q_gen = Q_inj + Q_load(|V|); without voltage-dependent load Q_load=0.
        # Collect violators first, then swap either all (top_k=None) or the
        # top-k heaviest ones.
        q_gen = S_calc.imag + (q_load_at_v if q_load_at_v is not None else 0.0)
        violators = q_limit_violations(q_gen, q_min, q_max, pv, tol=q_lim_tol)

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
        if not np.isnan(qmax) and qlocked == qmax and vk > vk_set + V_SET_DEADBAND:
            pq_set.discard(k)
            pv_set.add(k)
            locked_new[k] = np.nan
            actions.append(QLimAction(bus_idx=k, direction="pq->pv_qmax", q_value=vk))
        elif not np.isnan(qmin) and qlocked == qmin and vk < vk_set - V_SET_DEADBAND:
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
        locked_lim=locked_new,
        actions=actions,
        changed=len(actions) > 0,
        bus_q_gen_new=bus_q_gen_new,
    )
