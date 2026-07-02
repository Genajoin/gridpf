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

from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, get_args

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gridpf.algebra.sbus import build_sbus, classify_buses, compute_sbus, q_load_at
from gridpf.algebra.ybus import build_ybus
from gridpf.contract.types import BASE_MVA, Method, PFInput, PFOptions, PFResult
from gridpf.solvers.dc_pf import dc_powerflow
from gridpf.solvers.gauss_seidel import gauss_seidel
from gridpf.solvers.newton_raphson import NRResult, newton_raphson
from gridpf.solvers.q_lims import enforce_q_limits, q_limit_violations


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


# Extra slack on top of q_lim_tol when *reporting* violations: buses that
# enforcement pinned exactly at their limit must not be counted as violators
# because of the final NR residual.
_Q_VIOLATION_EPS = 1e-6

# Final mismatch above this norm reads as a diverged (collapsing) voltage
# solution rather than a mere max-iterations stall.
_VOLTAGE_COLLAPSE_MISMATCH = 1e3


def _has_orphan_component(net: PFInput) -> bool:
    """Есть ли связная компонента без slack-узла среди активных узлов?"""
    n = net.n_bus
    # Build the undirected adjacency over from_idx/to_idx branches as a sparse
    # graph and label every bus with its connected component. With no edges each
    # bus is its own component, which the labelling handles naturally.
    adjacency = coo_matrix(
        (np.ones(net.n_branch, dtype=np.int8), (net.from_idx, net.to_idx)),
        shape=(n, n),
    )
    _, labels = connected_components(adjacency, directed=False)
    slack_labels = set(labels[net.bus_type == 2].tolist())
    # Orphaned iff at least one bus lives in a component that holds no slack bus.
    return bool(np.any(~np.isin(labels, list(slack_labels))))


def _flat_start(net: PFInput) -> np.ndarray:
    """Сформировать flat-старт: ``|V|=1, δ=0`` для всех шин.

    Заданные модули/углы управляемых узлов накладываются отдельно через
    :func:`_apply_setpoints`.
    """
    n = net.n_bus
    Vm = np.ones(n, dtype=np.float64)
    Va = np.zeros(n, dtype=np.float64)
    result: np.ndarray = Vm * np.exp(1j * Va)
    return result


def _apply_setpoints(net: PFInput, V0: np.ndarray) -> np.ndarray:
    """Наложить заданные модули/углы управляемых узлов на ``V0``.

    Читает материализованные адаптером ``net.bus_v_set`` (|V| p.u.) и
    ``net.bus_va_set`` (угол, рад): где значение не ``NaN`` — оно замещает
    соответствующую компоненту ``V0``. Поля ``None`` → старт не меняется.
    """
    Vm = np.abs(V0).copy()
    Va = np.angle(V0).copy()
    if net.bus_v_set is not None:
        mask = ~np.isnan(net.bus_v_set)
        Vm[mask] = net.bus_v_set[mask]
    if net.bus_va_set is not None:
        mask = ~np.isnan(net.bus_va_set)
        Va[mask] = net.bus_va_set[mask]
    result: np.ndarray = Vm * np.exp(1j * Va)
    return result


@dataclass
class _NRState:
    """Voltage / iteration / convergence accumulator across solver passes.

    Every NR pass in the engine folds its outcome in through :meth:`absorb`,
    so the "run solver, unpack four fields" boilerplate lives in one place.
    """

    V: np.ndarray
    iters_nr: int = 0
    mismatch: float = field(default=float("nan"))
    converged: bool = False

    def absorb(self, res: NRResult) -> None:
        """Fold one solver pass into the accumulated state."""
        self.V = res.V
        self.iters_nr += res.iterations
        self.mismatch = res.mismatch_max
        self.converged = res.converged


@dataclass
class _Classification:
    """Bus classification plus the state the Q-limit loop swaps alongside it.

    A PV→PQ swap changes ``pv``/``pq``/``Sbus``/``locked_lim`` and (with
    voltage-dependent load) ``net.bus_q_gen`` together; bundling them makes
    snapshot/rollback a single-object operation instead of five parallel
    variables.
    """

    pv: np.ndarray
    pq: np.ndarray
    Sbus: np.ndarray
    locked_lim: np.ndarray
    net: PFInput

    def restore(self, snap: _Classification) -> None:
        """Roll back to a previously taken snapshot."""
        self.pv = snap.pv
        self.pq = snap.pq
        self.Sbus = snap.Sbus
        self.locked_lim = snap.locked_lim
        self.net = snap.net


def _run_nr(
    state: _NRState,
    cls: _Classification,
    Ybus: csr_matrix,
    ref: np.ndarray,
    *,
    options: PFOptions,
    use_load: bool,
) -> None:
    """One Newton-Raphson pass from ``state.V`` under the current classification."""
    state.absorb(
        newton_raphson(
            Ybus,
            cls.Sbus,
            state.V,
            ref,
            cls.pv,
            cls.pq,
            tol=options.tol,
            max_iter=options.max_iter_nr,
            net=cls.net,
            voltage_dependent_load=use_load,
        )
    )


def _nr_with_q_limits(
    state: _NRState,
    cls: _Classification,
    Ybus: csr_matrix,
    ref: np.ndarray,
    *,
    options: PFOptions,
    use_load: bool,
    can_enforce: bool,
    pv_original: np.ndarray,
    v_set_arr: np.ndarray,
) -> tuple[int, PFInput]:
    """Newton-Raphson plus the outer PV→PQ enforcement loop.

    Mutates ``state`` and ``cls`` in place. Without ``can_enforce`` this is
    exactly one NR pass. Returns ``(q_lim_swaps, pre_swap_net)`` where
    ``pre_swap_net`` is the network snapshot taken before the last committed
    swap — the soft fallback uses it to restore ``bus_q_gen``.
    """
    q_min = cls.net.bus_q_min
    q_max = cls.net.bus_q_max
    snap = _dc_replace(cls)
    outer_done = False
    q_lim_swaps = 0

    for _swap_iter in range(options.max_q_lim_swaps + 1):
        _run_nr(state, cls, Ybus, ref, options=options, use_load=use_load)

        if not can_enforce:
            break
        if not state.converged:
            # NR не сошёлся при текущей классификации. Откатываем последнее
            # переключение и пробуем добить решение без него — типично
            # «overshoot» при массовом swap'е делает следующий NR
            # неустойчивым.
            cls.restore(snap)
            _run_nr(state, cls, Ybus, ref, options=options, use_load=use_load)
            break

        assert q_min is not None and q_max is not None  # защищено can_enforce
        qlim_res = enforce_q_limits(
            Ybus,
            state.V,
            cls.Sbus,
            cls.pv,
            cls.pq,
            q_min,
            q_max,
            v_set_arr,
            cls.locked_lim,
            pv_original=pv_original,
            allow_pq_to_pv=options.allow_pq_to_pv,
            top_k=options.q_lim_top_k,
            q_lim_tol=options.q_lim_tol,
            # net передаётся ВСЕГДА: генераторная семантика лимитов
            # (Q_gen = Q_calc + Q_load) не зависит от активности СХН — при
            # неактивной СХН Q_load константна (= bus_q_load), но вычитать
            # её обязательно (см. enforce_q_limits).
            net=cls.net,
            voltage_dependent_load=use_load,
        )
        if not qlim_res.changed:
            outer_done = True
            break

        # Сохраняем «предыдущее хорошее» состояние перед коммитом swap'а
        snap = _dc_replace(cls)
        cls.pv = qlim_res.pv
        cls.pq = qlim_res.pq
        cls.locked_lim = qlim_res.locked_lim
        # При активной СХН: фиксируем bus_q_gen на лимите, далее Sbus
        # пересчитывается compute_sbus каждую NR-итерацию. Без СХН —
        # legacy: Sbus.imag = Q_lim напрямую.
        if use_load and qlim_res.bus_q_gen_new is not None:
            cls.net = _dc_replace(cls.net, bus_q_gen=qlim_res.bus_q_gen_new)
            cls.Sbus = compute_sbus(cls.net, state.V, voltage_dependent=True)
        else:
            cls.Sbus = qlim_res.Sbus
        q_lim_swaps += len(qlim_res.actions)
        state.converged = False  # требуется ещё один прогон NR с новой классификацией

    # Если outer-loop достиг лимита (а не отработал break по changed=False),
    # делаем финальный NR с уже коммитнутой классификацией — даёт шанс
    # «дотянуть» сходимость на тех моделях, где enforcement-цикл сошёлся
    # фактически, но формально не успел подтвердиться.
    if can_enforce and not outer_done and not state.converged:
        _run_nr(state, cls, Ybus, ref, options=options, use_load=use_load)

    return q_lim_swaps, snap.net


def _soft_fallback(
    state: _NRState,
    cls: _Classification,
    Ybus: csr_matrix,
    *,
    options: PFOptions,
    use_load: bool,
    pre_swap_net: PFInput,
) -> None:
    """Retry with the *original* classification when enforcement failed.

    Guarantees ``enforce_q_lims=True`` is never worse than the baseline run;
    remaining violations are reported via ``PFResult.q_violations``. Mutates
    ``state``/``cls`` only when the fallback NR converges (a failed fallback
    leaves the previous best state, iteration counts included, untouched).
    """
    # Восстанавливаем исходную классификацию узлов и оригинальный
    # bus_q_gen (без swap-фиксаций); СХН остаётся включённой.
    net_orig = (
        _dc_replace(cls.net, bus_q_gen=pre_swap_net.bus_q_gen)
        if pre_swap_net is not cls.net
        else cls.net
    )
    ref_orig, pv_orig_arr, pq_orig_arr = classify_buses(net_orig.bus_type)
    sbus_orig = (
        compute_sbus(net_orig, _flat_start(net_orig), voltage_dependent=True)
        if use_load
        else build_sbus(net_orig)
    )
    # Для устойчивости — повторный flat + GS warm-start.
    V_fb = _apply_setpoints(net_orig, _flat_start(net_orig))
    if options.method in ("gs", "gs+nr"):
        gs_fb = gauss_seidel(
            Ybus,
            sbus_orig,
            V_fb,
            ref_orig,
            pv_orig_arr,
            pq_orig_arr,
            tol=options.gs_tol,
            max_iter=options.max_iter_gs,
            net=net_orig,
            voltage_dependent_load=use_load,
        )
        V_fb = gs_fb.V
    nr_fb = newton_raphson(
        Ybus,
        sbus_orig,
        V_fb,
        ref_orig,
        pv_orig_arr,
        pq_orig_arr,
        tol=options.tol,
        max_iter=options.max_iter_nr,
        net=net_orig,
        voltage_dependent_load=use_load,
    )
    if nr_fb.converged:
        state.absorb(nr_fb)
        cls.net = net_orig
        cls.pv = pv_orig_arr


def _dc_fallback(
    state: _NRState,
    cls: _Classification,
    Ybus: csr_matrix,
    *,
    options: PFOptions,
    use_load: bool,
) -> None:
    """DC warm-start retry: linearized angles, then one more NR pass.

    Mutates ``state``/``cls`` only when the retry converges.
    """
    ref_d, pv_d, pq_d = classify_buses(cls.net.bus_type)
    sbus_d = build_sbus(cls.net)
    delta_dc = dc_powerflow(
        cls.net,
        P_inj=sbus_d.real,
        ref=ref_d,
        pv=pv_d,
        pq=pq_d,
    )
    # Стартовое V для NR: модули как в flat+setpoints, углы из DC.
    V_dc = _apply_setpoints(cls.net, _flat_start(cls.net))
    V_dc = np.abs(V_dc) * np.exp(1j * delta_dc)
    nr_dc = newton_raphson(
        Ybus,
        sbus_d,
        V_dc,
        ref_d,
        pv_d,
        pq_d,
        tol=options.tol,
        max_iter=options.max_iter_nr,
        net=cls.net,
        voltage_dependent_load=use_load,
    )
    if nr_dc.converged:
        state.absorb(nr_dc)
        cls.pv = pv_d


def _branch_flows(
    net: PFInput, V: np.ndarray, Yf: csr_matrix, Yt: csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    """Branch power flows in MVA (``S_base × p.u.``) for both branch ends."""
    if net.n_branch == 0:
        empty = np.empty(0, dtype=np.complex128)
        return empty, empty
    s_from = V[net.from_idx] * np.conj(Yf @ V) * BASE_MVA
    s_to = V[net.to_idx] * np.conj(Yt @ V) * BASE_MVA
    return s_from, s_to


def _count_q_violations(
    net: PFInput,
    V: np.ndarray,
    Ybus: csr_matrix,
    pv_original: np.ndarray,
    *,
    q_lim_tol: float,
    use_load: bool,
) -> int:
    """Count originally-PV buses violating generator Q-limits at the solution.

    Generator semantics, shared with :func:`enforce_q_limits`: the limit is
    compared against ``Q_gen = Q_inj + Q_load(|V|)``, not the net injection —
    otherwise a loaded generator bus reports a false violation. The deadband
    is widened by :data:`_Q_VIOLATION_EPS` so reporting stays consistent with
    what enforcement pinned at the limit.
    """
    q_min, q_max = net.bus_q_min, net.bus_q_max
    assert q_min is not None and q_max is not None  # guarded by can_enforce
    S_calc = V * np.conj(Ybus @ V)
    q_gen = S_calc.imag + q_load_at(net, V, voltage_dependent=use_load)
    return len(
        q_limit_violations(q_gen, q_min, q_max, pv_original, tol=q_lim_tol + _Q_VIOLATION_EPS)
    )


def _classify_failure(
    net: PFInput,
    mismatch: float,
    *,
    implausible_v_nodes: int,
    q_violations_reported: bool,
) -> str:
    """Map a non-converged final state to a short ``failure_reason`` code."""
    if implausible_v_nodes:
        return "implausible_voltage"
    if _has_orphan_component(net):
        return "no_slack_component"
    if not np.isfinite(mismatch):
        return "singular_jacobian"
    if mismatch > _VOLTAGE_COLLAPSE_MISMATCH:
        return "voltage_collapse"
    if q_violations_reported:
        return "infeasible_q_lims"
    return "max_iter_reached"


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

    Note:
        Сингулярный якобиан Newton НЕ бросает исключение: scipy на вырожденной
        матрице возвращает NaN-вектор, итерация прерывается с последним конечным
        ``V`` → ``PFResult(converged=False, failure_reason="singular_jacobian")``.
    """
    method = options.method
    allowed_methods = get_args(Method)
    if method not in allowed_methods:
        allowed = ", ".join(repr(m) for m in allowed_methods)
        raise ValueError(f"Неизвестный method={method!r}; ожидался один из: {allowed}.")

    Ybus, Yf, Yt = build_ybus(net)
    Sbus = build_sbus(net)
    ref, pv, pq = classify_buses(net.bus_type)

    # Стартовое приближение: тёплый старт init_v (выровнен адаптером по
    # net.bus_ids) либо flat. None → flat (бит-в-бит).
    # Setpoints PV/slack клампятся ПОВЕРХ старта (Rule №4): заданный модуль/угол
    # всегда побеждает тёплый старт на управляемых узлах.
    V0 = init_v if init_v is not None else _flat_start(net)
    V0 = _apply_setpoints(net, V0)

    # Начальные параметры enforce_q_lims (если выключено — массивы не используются).
    pv_original = pv.copy()
    v_set_arr = np.full(net.n_bus, np.nan, dtype=np.float64)
    v_set_arr[pv] = np.abs(V0[pv])
    can_enforce = options.enforce_q_lims and net.bus_q_min is not None and net.bus_q_max is not None

    # СХН активна, если: пользователь не отключил, в сети есть нетривиальные
    # коэффициенты и доступны базовые поля. Совместима с enforce_q_lims:
    # внешний цикл swap'ов фиксирует bus_q_gen на Q-лимите, дальнейшие NR
    # пересчитывают Sbus через compute_sbus(net, V).
    use_load_v = options.use_load_voltage_dependency and net.has_voltage_dependent_load
    # Перестраиваем Sbus с учётом СХН на flat-старте — чтобы GS не стартовал
    # с устаревшей константой.
    if use_load_v:
        Sbus = compute_sbus(net, V0, voltage_dependent=True)

    state = _NRState(V=V0)
    cls = _Classification(
        pv=pv,
        pq=pq,
        Sbus=Sbus,
        locked_lim=np.full(net.n_bus, np.nan, dtype=np.float64),
        net=net,
    )
    iters_gs = 0
    q_lim_swaps = 0

    # GS warm-start выполняется один раз — переключения PV/PQ затрагивают только NR-фазу.
    if method in ("gs", "gs+nr"):
        gs_res = gauss_seidel(
            Ybus,
            Sbus,
            V0,
            ref,
            pv,
            pq,
            tol=options.tol if method == "gs" else options.gs_tol,
            max_iter=options.max_iter_gs,
            net=net,
            voltage_dependent_load=use_load_v,
        )
        state.V = gs_res.V
        iters_gs = gs_res.iterations
        state.mismatch = gs_res.mismatch_max
        if method == "gs":
            state.converged = gs_res.converged

    # Внешний цикл по Q-лимитам. Без enforce_q_lims — ровно одна итерация NR.
    if method in ("nr", "gs+nr"):
        q_lim_swaps, pre_swap_net = _nr_with_q_limits(
            state,
            cls,
            Ybus,
            ref,
            options=options,
            use_load=use_load_v,
            can_enforce=can_enforce,
            pv_original=pv_original,
            v_set_arr=v_set_arr,
        )
        # Soft-fallback: если ничего не помогло, отказываемся от enforcement
        # и возвращаемся к классификации **без переключений**.
        if can_enforce and not state.converged:
            _soft_fallback(
                state,
                cls,
                Ybus,
                options=options,
                use_load=use_load_v,
                pre_swap_net=pre_swap_net,
            )

    # DC warm-start fallback: если NR-расчёт (с/без enforcement) разошёлся,
    # пробуем линеаризованное DC-приближение для углов и стартуем NR ещё раз.
    if options.dc_fallback and not state.converged and method in ("nr", "gs+nr"):
        _dc_fallback(state, cls, Ybus, options=options, use_load=use_load_v)

    V = state.V
    s_from, s_to = _branch_flows(cls.net, V, Yf, Yt)

    q_violations = 0
    if can_enforce and state.converged:
        q_violations = _count_q_violations(
            cls.net,
            V,
            Ybus,
            pv_original,
            q_lim_tol=options.q_lim_tol,
            use_load=use_load_v,
        )

    # Sanity-гейт правдоподобия: NR может численно сойтись (mismatch < tol)
    # в нижнюю ветвь PV-кривой (|V| ~ 0.1–0.3 p.u. при несогласованных
    # инжекциях) — физически это несошедшийся режим.
    implausible_v_nodes = 0
    if state.converged and options.v_plausible_range is not None:
        v_lo, v_hi = options.v_plausible_range
        vm_final = np.abs(V)
        implausible_v_nodes = int(np.count_nonzero((vm_final < v_lo) | (vm_final > v_hi)))
        if implausible_v_nodes:
            state.converged = False

    failure_reason = ""
    if not state.converged:
        failure_reason = _classify_failure(
            cls.net,
            state.mismatch,
            implausible_v_nodes=implausible_v_nodes,
            q_violations_reported=can_enforce and q_violations > 0,
        )

    return PFResult(
        converged=state.converged,
        iterations_gs=iters_gs,
        iterations_nr=state.iters_nr,
        V=V,
        bus_ids=cls.net.bus_ids,
        S_from=s_from,
        S_to=s_to,
        mismatch_max=state.mismatch,
        method=method,
        q_lim_swaps=q_lim_swaps,
        q_violations=q_violations,
        failure_reason=failure_reason,
        implausible_v_nodes=implausible_v_nodes,
        voltage_dependent_load_active=use_load_v,
    )
