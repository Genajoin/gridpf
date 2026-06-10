"""Ядро движка установившегося режима — внешний цикл (model-free).

Извлечено из адаптерного ``solve_pf`` без какой-либо привязки к классу модели
сети: на вход — собранный адаптером :class:`~gridpf.contract.types.PFInput`
(opaque p.u.) и опциональный тёплый старт ``init_v``; на выход —
:class:`~gridpf.contract.types.PFResult`. Запись результата обратно во внешнюю
модель — забота адаптера, движок ничего не пишет.

Логика: GS warm-start → Newton-Raphson + внешний Q-lim цикл → DC-fallback →
soft-fallback к классификации без переключений. Setpoints управляемых узлов
(``net.bus_v_set`` / ``net.bus_va_set``) клампятся ПОВЕРХ стартового
приближения (Rule №4).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

import numpy as np

from gridpf.algebra.sbus import build_sbus, classify_buses, compute_sbus
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.types import BASE_MVA, PFInput, PFOptions, PFResult
from gridpf.solvers.dc_pf import dc_powerflow
from gridpf.solvers.gauss_seidel import gauss_seidel
from gridpf.solvers.newton_raphson import newton_raphson
from gridpf.solvers.q_lims import enforce_q_limits


def _has_orphan_component(network_pu: PFInput) -> bool:
    """Есть ли связная компонента без slack-узла среди активных узлов?"""
    from collections import deque

    n = network_pu.n_bus
    adj: list[list[int]] = [[] for _ in range(n)]
    for f, t in zip(network_pu.from_idx.tolist(), network_pu.to_idx.tolist(), strict=True):
        adj[f].append(t)
        adj[t].append(f)
    slack_set = set(np.where(network_pu.bus_type == 2)[0].tolist())
    seen = set(slack_set)
    q: deque[int] = deque(slack_set)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) < n  # есть узлы недостижимые от slack


def _flat_start(network_pu: PFInput) -> np.ndarray:
    """Сформировать flat-старт: ``|V|=1, δ=0`` для всех шин.

    Заданные модули/углы управляемых узлов накладываются отдельно через
    :func:`_apply_setpoints`.
    """
    n = network_pu.n_bus
    Vm = np.ones(n, dtype=np.float64)
    Va = np.zeros(n, dtype=np.float64)
    result: np.ndarray = Vm * np.exp(1j * Va)
    return result


def _apply_setpoints(network_pu: PFInput, V0: np.ndarray) -> np.ndarray:
    """Наложить заданные модули/углы управляемых узлов на ``V0``.

    Читает материализованные адаптером ``net.bus_v_set`` (|V| p.u.) и
    ``net.bus_va_set`` (угол, рад): где значение не ``NaN`` — оно замещает
    соответствующую компоненту ``V0``. Поля ``None`` → старт не меняется.
    """
    Vm = np.abs(V0).copy()
    Va = np.angle(V0).copy()
    if network_pu.bus_v_set is not None:
        mask = ~np.isnan(network_pu.bus_v_set)
        Vm[mask] = network_pu.bus_v_set[mask]
    if network_pu.bus_va_set is not None:
        mask = ~np.isnan(network_pu.bus_va_set)
        Va[mask] = network_pu.bus_va_set[mask]
    result: np.ndarray = Vm * np.exp(1j * Va)
    return result


def run_powerflow(
    net: PFInput,
    options: PFOptions,
    *,
    init_v: np.ndarray | None = None,
) -> PFResult:
    """Рассчитать установившийся режим для ``net`` (model-free).

    Args:
        net: opaque p.u.-представление сети.
        options: опции движка.
        init_v: ``(n_bus,)`` complex — тёплый старт (например, из результатов
            прошлого прогона, выровненный по ``net.bus_ids``). ``None`` →
            flat-старт (``|V|=1, δ=0``), бит-в-бит с холодным запуском.

    Returns:
        :class:`PFResult` со всеми итоговыми величинами. Запись обратно —
        не делается (забота адаптера).

    Raises:
        ValueError: если метод неизвестен или в сети нет slack-узла.
        RuntimeError: если якобиан Newton сингулярен.
    """
    method = options.method
    if method not in ("gs+nr", "nr", "gs"):
        raise ValueError(f"Неизвестный method={method!r}; ожидался 'gs+nr', 'nr' или 'gs'.")
    tol = options.tol
    max_iter_gs = options.max_iter_gs
    max_iter_nr = options.max_iter_nr
    gs_tol = options.gs_tol
    enforce_q_lims = options.enforce_q_lims
    max_q_lim_swaps = options.max_q_lim_swaps
    allow_pq_to_pv = options.allow_pq_to_pv
    q_lim_top_k = options.q_lim_top_k
    dc_fallback = options.dc_fallback
    use_load_voltage_dependency = options.use_load_voltage_dependency

    network_pu = net
    ybus, yf, yt = build_ybus(network_pu)
    sbus = build_sbus(network_pu)
    ref, pv, pq = classify_buses(network_pu.bus_type)

    # Стартовое приближение: тёплый старт init_v (выровнен адаптером по
    # network_pu.bus_ids) либо flat. None → flat (бит-в-бит).
    V0 = init_v if init_v is not None else _flat_start(network_pu)
    # Setpoints PV/slack клампятся ПОВЕРХ старта (Rule №4): заданный модуль/угол
    # всегда побеждает тёплый старт на управляемых узлах.
    V0 = _apply_setpoints(network_pu, V0)

    # Начальные параметры enforce_q_lims (если выключено — массивы не используются).
    pv_original = pv.copy()
    v_set_arr = np.full(network_pu.n_bus, np.nan, dtype=np.float64)
    v_set_arr[pv] = np.abs(V0[pv])
    locked_lim = np.full(network_pu.n_bus, np.nan, dtype=np.float64)
    q_min = network_pu.bus_q_min
    q_max = network_pu.bus_q_max
    can_enforce = enforce_q_lims and q_min is not None and q_max is not None

    iters_gs = 0
    iters_nr = 0
    q_lim_swaps = 0
    V = V0
    converged = False
    mismatch = float("nan")

    # СХН активна, если: пользователь не отключил, в сети есть нетривиальные
    # коэффициенты и доступны базовые поля. Совместима с enforce_q_lims:
    # внешний цикл swap'ов фиксирует bus_q_gen на Q-лимите, дальнейшие NR
    # пересчитывают Sbus через compute_sbus(network_pu, V).
    use_load_v = use_load_voltage_dependency and network_pu.has_voltage_dependent_load
    # Перестраиваем Sbus с учётом СХН на flat-старте — чтобы GS не стартовал
    # с устаревшей константой.
    if use_load_v:
        sbus = compute_sbus(network_pu, V0, voltage_dependent=True)

    # GS warm-start выполняется один раз — переключения PV/PQ затрагивают только NR-фазу.
    if method in ("gs", "gs+nr"):
        gs_local_tol = tol if method == "gs" else gs_tol
        gs_res = gauss_seidel(
            ybus,
            sbus,
            V0,
            ref,
            pv,
            pq,
            tol=gs_local_tol,
            max_iter=max_iter_gs,
            network_pu=network_pu,
            voltage_dependent_load=use_load_v,
        )
        V = gs_res.V
        iters_gs = gs_res.iterations
        mismatch = gs_res.mismatch_max
        if method == "gs":
            converged = gs_res.converged

    # Внешний цикл по Q-лимитам. Без enforce_q_lims — ровно одна итерация NR.
    if method in ("nr", "gs+nr"):
        last_swap_pv = pv.copy()
        last_swap_pq = pq.copy()
        last_swap_sbus = sbus.copy()
        last_swap_locked = locked_lim.copy()
        last_swap_net = network_pu  # для отката bus_q_gen
        outer_done = False
        for _swap_iter in range(max_q_lim_swaps + 1):
            nr_res = newton_raphson(
                ybus,
                sbus,
                V,
                ref,
                pv,
                pq,
                tol=tol,
                max_iter=max_iter_nr,
                network_pu=network_pu,
                voltage_dependent_load=use_load_v,
            )
            V = nr_res.V
            iters_nr += nr_res.iterations
            mismatch = nr_res.mismatch_max
            converged = nr_res.converged

            if not can_enforce:
                break
            if not converged:
                # NR не сошёлся при текущей классификации. Откатываем последнее
                # переключение и пробуем добить решение без него — типично
                # «overshoot» при массовом swap'е делает следующий NR
                # неустойчивым.
                pv = last_swap_pv
                pq = last_swap_pq
                sbus = last_swap_sbus
                locked_lim = last_swap_locked
                network_pu = last_swap_net
                nr_res = newton_raphson(
                    ybus,
                    sbus,
                    V,
                    ref,
                    pv,
                    pq,
                    tol=tol,
                    max_iter=max_iter_nr,
                    network_pu=network_pu,
                    voltage_dependent_load=use_load_v,
                )
                V = nr_res.V
                iters_nr += nr_res.iterations
                mismatch = nr_res.mismatch_max
                converged = nr_res.converged
                break

            assert q_min is not None and q_max is not None  # защищено can_enforce
            qlim_res = enforce_q_limits(
                ybus,
                V,
                sbus,
                pv,
                pq,
                q_min,
                q_max,
                v_set_arr,
                locked_lim,
                pv_original=pv_original,
                allow_pq_to_pv=allow_pq_to_pv,
                top_k=q_lim_top_k,
                # network_pu передаётся ВСЕГДА: генераторная семантика лимитов
                # (Q_gen = Q_calc + Q_load) не зависит от активности СХН — при
                # неактивной СХН Q_load константна (= bus_q_load), но вычитать
                # её обязательно (см. enforce_q_limits).
                network_pu=network_pu,
                voltage_dependent_load=use_load_v,
            )
            if not qlim_res.changed:
                outer_done = True
                break
            # Сохраняем «предыдущее хорошее» состояние перед коммитом swap'а
            last_swap_pv = pv
            last_swap_pq = pq
            last_swap_sbus = sbus
            last_swap_locked = locked_lim
            last_swap_net = network_pu
            pv = qlim_res.pv
            pq = qlim_res.pq
            locked_lim = qlim_res.locked_lim
            # При активной СХН: фиксируем bus_q_gen на лимите, далее sbus
            # пересчитывается compute_sbus каждую NR-итерацию. Без СХН —
            # legacy: Sbus.imag = Q_lim напрямую.
            if use_load_v and qlim_res.bus_q_gen_new is not None:
                network_pu = _dc_replace(network_pu, bus_q_gen=qlim_res.bus_q_gen_new)
                sbus = compute_sbus(network_pu, V, voltage_dependent=True)
            else:
                sbus = qlim_res.Sbus
            q_lim_swaps += len(qlim_res.actions)
            converged = False  # требуется ещё один прогон NR с новой классификацией

        # Если outer-loop достиг лимита (а не отработал break по changed=False),
        # делаем финальный NR с уже коммитнутой классификацией — даёт шанс
        # «дотянуть» сходимость на тех моделях, где enforcement-цикл сошёлся
        # фактически, но формально не успел подтвердиться.
        if can_enforce and not outer_done and not converged:
            nr_res = newton_raphson(
                ybus,
                sbus,
                V,
                ref,
                pv,
                pq,
                tol=tol,
                max_iter=max_iter_nr,
                network_pu=network_pu,
                voltage_dependent_load=use_load_v,
            )
            V = nr_res.V
            iters_nr += nr_res.iterations
            mismatch = nr_res.mismatch_max
            converged = nr_res.converged

        # Soft-fallback: если ничего не помогло, отказываемся от enforcement
        # и возвращаемся к классификации **без переключений**. Гарантирует,
        # что enforce_q_lims=True никогда не хуже baseline; нарушения
        # репортуются через PFResult.q_violations.
        if can_enforce and not converged:
            # Восстанавливаем исходную классификацию узлов и оригинальный
            # bus_q_gen (без swap-фиксаций); СХН остаётся включённой.
            net_orig = (
                _dc_replace(network_pu, bus_q_gen=last_swap_net.bus_q_gen)
                if last_swap_net is not network_pu
                else network_pu
            )
            ref_orig, pv_orig_arr, pq_orig_arr = classify_buses(net_orig.bus_type)
            sbus_orig = (
                compute_sbus(net_orig, _flat_start(net_orig), voltage_dependent=True)
                if use_load_v
                else build_sbus(net_orig)
            )
            # Для устойчивости — повторный flat + GS warm-start.
            V_fb = _flat_start(net_orig)
            V_fb = _apply_setpoints(net_orig, V_fb)
            if method in ("gs", "gs+nr"):
                gs_fb = gauss_seidel(
                    ybus,
                    sbus_orig,
                    V_fb,
                    ref_orig,
                    pv_orig_arr,
                    pq_orig_arr,
                    tol=gs_tol,
                    max_iter=max_iter_gs,
                    network_pu=net_orig,
                    voltage_dependent_load=use_load_v,
                )
                V_fb = gs_fb.V
            nr_fb = newton_raphson(
                ybus,
                sbus_orig,
                V_fb,
                ref_orig,
                pv_orig_arr,
                pq_orig_arr,
                tol=tol,
                max_iter=max_iter_nr,
                network_pu=net_orig,
                voltage_dependent_load=use_load_v,
            )
            if nr_fb.converged:
                V = nr_fb.V
                iters_nr += nr_fb.iterations
                mismatch = nr_fb.mismatch_max
                converged = True
                network_pu = net_orig
                pv = pv_orig_arr  # для подсчёта q_violations

    # DC warm-start fallback: если NR-расчёт (с/без enforcement) разошёлся,
    # пробуем линеаризованное DC-приближение для углов и стартуем NR ещё раз.
    if dc_fallback and not converged and method in ("nr", "gs+nr"):
        ref_d, pv_d, pq_d = classify_buses(network_pu.bus_type)
        sbus_d = build_sbus(network_pu)
        delta_dc = dc_powerflow(
            n_bus=network_pu.n_bus,
            from_idx=network_pu.from_idx,
            to_idx=network_pu.to_idx,
            branch_x=network_pu.branch_x,
            tap_ratio=network_pu.tap_ratio,
            P_inj=sbus_d.real,
            ref=ref_d,
            pv=pv_d,
            pq=pq_d,
        )
        # Стартовое V для NR: модули как в flat+setpoints, углы из DC.
        V_dc = _flat_start(network_pu)
        V_dc = _apply_setpoints(network_pu, V_dc)
        Vm_dc = np.abs(V_dc)
        V_dc = Vm_dc * np.exp(1j * delta_dc)
        nr_dc = newton_raphson(
            ybus,
            sbus_d,
            V_dc,
            ref_d,
            pv_d,
            pq_d,
            tol=tol,
            max_iter=max_iter_nr,
            network_pu=network_pu,
            voltage_dependent_load=use_load_v,
        )
        if nr_dc.converged:
            V = nr_dc.V
            iters_nr += nr_dc.iterations
            mismatch = nr_dc.mismatch_max
            converged = True
            pv = pv_d

    # Потоки ветвей в МВА (S_base × p.u.)
    if network_pu.n_branch > 0:
        s_from = V[network_pu.from_idx] * np.conj(yf @ V) * BASE_MVA
        s_to = V[network_pu.to_idx] * np.conj(yt @ V) * BASE_MVA
    else:
        s_from = np.empty(0, dtype=np.complex128)
        s_to = np.empty(0, dtype=np.complex128)

    # Подсчёт PV-узлов с нарушением Q-лимитов в финальном решении.
    # Q-лимит сравнивается с Q_gen = Q_inj + Q_load (генераторная семантика,
    # как в enforce_q_limits), а не с сетевой Q_inj — иначе мы ложно репортуем
    # нарушение из-за нагрузки на узле. При активной СХН Q_load зависит от
    # |V| (полином), иначе — константа bus_q_load.
    q_violations = 0
    if can_enforce and converged:
        I_bus = ybus @ V
        S_calc_final = V * np.conj(I_bus)
        if network_pu.bus_q_load is None:
            q_load_fin = np.zeros(network_pu.n_bus)
        elif use_load_v:
            Vm_fin = np.abs(V)
            b0 = network_pu.bus_q_b0
            b1 = network_pu.bus_q_b1
            b2 = network_pu.bus_q_b2
            assert b0 is not None and b1 is not None and b2 is not None
            q_load_fin = network_pu.bus_q_load * (b0 + b1 * Vm_fin + b2 * Vm_fin * Vm_fin)
        else:
            q_load_fin = np.asarray(network_pu.bus_q_load, dtype=np.float64)
        for k in pv_original.tolist():
            qk_gen = float(S_calc_final[k].imag) + float(q_load_fin[k])
            qmax_k = q_max[k] if q_max is not None else np.nan
            qmin_k = q_min[k] if q_min is not None else np.nan
            if (not np.isnan(qmax_k) and qk_gen > qmax_k + 1e-6) or (
                not np.isnan(qmin_k) and qk_gen < qmin_k - 1e-6
            ):
                q_violations += 1

    failure_reason = ""
    if not converged:
        if _has_orphan_component(network_pu):
            failure_reason = "no_slack_component"
        elif np.isnan(mismatch) or not np.isfinite(mismatch):
            failure_reason = "singular_jacobian"
        elif mismatch > 1e3:
            failure_reason = "voltage_collapse"
        elif can_enforce and q_violations > 0:
            failure_reason = "infeasible_q_lims"
        else:
            failure_reason = "max_iter_reached"

    return PFResult(
        converged=converged,
        iterations_gs=iters_gs,
        iterations_nr=iters_nr,
        V=V,
        bus_ids=network_pu.bus_ids,
        S_from=s_from,
        S_to=s_to,
        mismatch_max=mismatch,
        method=method,
        q_lim_swaps=q_lim_swaps,
        q_violations=q_violations,
        failure_reason=failure_reason,
        voltage_dependent_load_active=use_load_v,
    )
