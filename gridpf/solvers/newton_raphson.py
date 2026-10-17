"""Newton-Raphson итератор для расчёта PF.

Базовая полярная формулировка без поддержки FACTS/VSC/TDPF (упрощённый
аналог ``pandapower/pypower/newtonpf.py``). Решение системы ``J · Δx = −F``
на каждой итерации делается через ``scipy.sparse.linalg.spsolve``.

Переменные:

* ``δ`` — углы PV ∪ PQ;
* ``|V|`` — модули PQ.

Slack — фиксирован; на PV модуль |V| фиксирован, корректируется только угол.

Сходимость — по ``∞``-норме небаланса
``F = [Re(mis[pvpq]); Im(mis[pq])]``, где
``mis = V · conj(Ybus · V) − Sbus``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from gridpf.algebra.jacobian import build_jacobian
from gridpf.algebra.sbus import compute_sbus, load_voltage_derivatives
from gridpf.solvers._common import (
    SolverResult,
    mismatch,
    residual_norm,
    residual_vector,
    resolve_use_load,
)
from gridpf.solvers.q_lims import InnerQLimits
from gridpf.solvers.step_limits import (
    StepLimits,
    branch_pairs,
    corridor_correction,
    expand_increment,
    step_scale,
)


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


NRResult = SolverResult


def _sbus_at(
    V: np.ndarray,
    sbus_const: np.ndarray,
    net: PFInput | None,
    voltage_dependent_load: bool,
) -> np.ndarray:
    """Вернуть Sbus с учётом СХН (если задан net и флаг)."""
    if net is None or not voltage_dependent_load:
        return sbus_const
    return compute_sbus(net, V, voltage_dependent=True)


def _evaluate(
    Ybus: csr_matrix,
    Vm: np.ndarray,
    Va: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    Sbus_const: np.ndarray,
    net: PFInput | None,
    voltage_dependent_load: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(V, F, ‖F‖∞)`` at the polar point ``(Vm, Va)``."""
    V = Vm * np.exp(1j * Va)
    Sbus_v = _sbus_at(V, Sbus_const, net, voltage_dependent_load)
    f = residual_vector(mismatch(Ybus, V, Sbus_v), pv, pq)
    return V, f, residual_norm(f)


def _try_step(
    Ybus: csr_matrix,
    Vm: np.ndarray,
    Va: np.ndarray,
    dx: np.ndarray,
    mu: float,
    pv: np.ndarray,
    pq: np.ndarray,
    Sbus_const: np.ndarray,
    net: PFInput | None,
    voltage_dependent_load: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Применить шаг ``μ·dx`` и вернуть новые ``(Vm, Va, V, F, ‖F‖∞)``."""
    n_pv = pv.size
    n_pq = pq.size
    Vm_new = Vm.copy()
    Va_new = Va.copy()
    if n_pv > 0:
        Va_new[pv] = Va[pv] + mu * dx[:n_pv]
    if n_pq > 0:
        Va_new[pq] = Va[pq] + mu * dx[n_pv : n_pv + n_pq]
        Vm_new[pq] = Vm[pq] + mu * dx[n_pv + n_pq :]
    V_new = Vm_new * np.exp(1j * Va_new)
    Sbus_new = _sbus_at(V_new, Sbus_const, net, voltage_dependent_load)
    mis = mismatch(Ybus, V_new, Sbus_new)
    f = residual_vector(mis, pv, pq)
    norm = residual_norm(f)
    return Vm_new, Va_new, V_new, f, norm


def newton_raphson(
    Ybus: csr_matrix,
    Sbus: np.ndarray,
    V0: np.ndarray,
    ref: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 30,
    step_control: bool = True,
    net: PFInput | None = None,
    voltage_dependent_load: bool = False,
    step_limits: StepLimits | None = None,
    q_ctrl: InnerQLimits | None = None,
) -> NRResult:
    """Прогнать Newton-Raphson до сходимости или ``max_iter``.

    Args:
        Ybus: ``(n, n)`` CSR.
        Sbus: ``(n,)`` complex — заданные инъекции (p.u.). Используются как
            константные, если ``voltage_dependent_load=False``. При включённой
            СХН служат стартовым приближением, но фактически каждую итерацию
            пересчитываются через ``compute_sbus(net, V)``.
        V0: ``(n,)`` complex — стартовое напряжение (например, после GS warm-start).
        ref: индексы slack-шин.
        pv: индексы PV-шин.
        pq: индексы PQ-шин.
        tol: целевая ∞-норма небаланса.
        max_iter: максимальное число итераций.
        step_control: если ``True`` — backtracking line search по μ при росте
            нормы небаланса (μ ← μ/2, до 5 раз). Расширяет область сходимости
            NR на сетях с большим начальным небалансом. По умолчанию ``True``.
        net: p.u.-представление сети; нужно при ``voltage_dependent_load=True``
            для пересчёта Sbus и поправки якобиана от СХН.
        voltage_dependent_load: учитывать СХН (полиномиальную зависимость
            нагрузки от ``|V|``). По умолчанию ``False`` для backward compat.
        step_limits: step-size limits and the post-step corridor (see
            :mod:`gridpf.solvers.step_limits`). ``None`` or all-``None``
            limits — unlimited step, bit-for-bit with the previous behavior.
        q_ctrl: Q-limit controller of ``q_lim_mode="inner"``. When given, it
            owns ``pv`` / ``pq`` / ``Sbus`` / ``net`` (the arguments are only
            the starting point), its :meth:`~InnerQLimits.step_check` runs
            after every step and its
            :meth:`~InnerQLimits.convergence_switch` gates convergence: a
            switch re-evaluates the residual with the new bus types instead
            of declaring convergence.

    Returns:
        ``NRResult`` с финальным напряжением, флагом сходимости, числом
        итераций и достигнутой нормой небаланса.
    """
    if max_iter < 0:
        raise ValueError(f"max_iter должен быть ≥ 0, получено {max_iter}")
    _ = ref  # slack учитывается через отсутствие в pv/pq; формальный параметр API.

    use_load = resolve_use_load(net, voltage_dependent_load)

    V = V0.astype(np.complex128, copy=True)
    Vm = np.abs(V)
    Va = np.angle(V)

    if q_ctrl is not None:
        pv, pq, Sbus, net = q_ctrl.pv, q_ctrl.pq, q_ctrl.Sbus, q_ctrl.net

    Sbus_eff = _sbus_at(V, Sbus, net, use_load)
    mis = mismatch(Ybus, V, Sbus_eff)
    f_vec = residual_vector(mis, pv, pq)
    norm_f = residual_norm(f_vec)

    if q_ctrl is not None and norm_f < tol and max_iter > 0:
        changed, released = q_ctrl.convergence_switch(V)
        if changed:
            pv, pq, Sbus, net = q_ctrl.pv, q_ctrl.pq, q_ctrl.Sbus, q_ctrl.net
            Vm[released] = q_ctrl.v_set[released]
            V, f_vec, norm_f = _evaluate(Ybus, Vm, Va, pv, pq, Sbus, net, use_load)

    if norm_f < tol or max_iter == 0:
        return NRResult(V=V, converged=norm_f < tol, iterations=0, mismatch_max=norm_f)

    scaling = step_limits is not None and step_limits.scaling_active
    corridor = step_limits is not None and step_limits.corridor_active
    pairs = branch_pairs(Ybus) if step_limits is not None and step_limits.needs_branches else None

    iteration = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        if use_load:
            assert net is not None
            dP_load, dQ_load = load_voltage_derivatives(net, V)
            dS_load_dVm: np.ndarray | None = (dP_load + 1j * dQ_load).astype(np.complex128)
        else:
            dS_load_dVm = None
        J = build_jacobian(Ybus, V, pv, pq, dS_load_dVm=dS_load_dVm)
        try:
            dx = spsolve(J, -f_vec)
        except RuntimeError:
            # SuperLU на точно-сингулярной матрице может БРОСИТЬ RuntimeError
            # («failed to factorize matrix») вместо NaN-вектора с warning —
            # оба исхода означают одно и то же. Приводим к NaN-пути ниже:
            # break с последним конечным V, mismatch=nan → singular_jacobian
            # и DC-fallback в _engine.
            dx = np.full_like(f_vec, np.nan)
        if not np.all(np.isfinite(dx)):
            # Сингулярный якобиан: scipy.spsolve на нём возвращает NaN-вектор
            # (с MatrixRankWarning), а НЕ исключение. Прерываем с последним
            # КОНЕЧНЫМ V; mismatch=nan → _engine классифицирует
            # singular_jacobian и активирует DC-fallback из живой точки
            # (вместо ~30 пустых итераций по NaN).
            norm_f = float("nan")
            break

        # Полный шаг μ=1 пробуем всегда (или μ=α при ограничении шага);
        # backtracking активен только если норма выросла И step_control=True.
        mu0 = 1.0
        if scaling:
            assert step_limits is not None
            dva, dvm = expand_increment(dx, V.size, pv, pq)
            mu0 = step_scale(Vm, dva, dvm, step_limits, pairs)
        mu_try = mu0
        Vm_try, Va_try, V_try, f_try, norm_try = _try_step(
            Ybus, Vm, Va, dx, mu0, pv, pq, Sbus, net, use_load
        )
        if step_control and norm_try > norm_f:
            mu = 0.5 * mu0
            descent = False
            for _bt in range(5):
                Vm_bt, Va_bt, V_bt, f_bt, norm_bt = _try_step(
                    Ybus, Vm, Va, dx, mu, pv, pq, Sbus, net, use_load
                )
                if norm_bt < norm_try:
                    Vm_try, Va_try, V_try, f_try, norm_try = (Vm_bt, Va_bt, V_bt, f_bt, norm_bt)
                    mu_try = mu
                if norm_bt < norm_f:
                    descent = True
                    break
                mu *= 0.5
            if not descent:
                # Бэктрекинг не нашёл спуска ниже norm_f → НЕ коммитим uphill-шаг.
                # Выходим из NR с последним ХОРОШИМ (Vm,Va,V,f_vec,norm_f) —
                # детерминированный неуспех из лучшей точки, здоровее DC-fallback.
                converged = False
                break

        if corridor and iteration > 1:
            # Post-step corridor: roll the whole applied step back so that the
            # worst outward violator lands on the boundary (no-op on iteration 1).
            assert step_limits is not None
            a = corridor_correction(Vm, Va, Vm_try, Va_try, step_limits, pairs)
            if a < 0.0:
                Vm_try, Va_try, V_try, f_try, norm_try = _try_step(
                    Ybus, Vm, Va, dx, mu_try * (1.0 + a), pv, pq, Sbus, net, use_load
                )

        Vm_prev, Va_prev = Vm, Va
        Vm, Va, V, f_vec, norm_f = Vm_try, Va_try, V_try, f_try, norm_try

        if q_ctrl is not None:
            released, undo = q_ctrl.step_check(V)
            if released.size:
                if undo:
                    Vm, Va = Vm_prev, Va_prev
                Vm = Vm.copy()
                Vm[released] = q_ctrl.v_set[released]
                pv, pq = q_ctrl.pv, q_ctrl.pq
                V, f_vec, norm_f = _evaluate(Ybus, Vm, Va, pv, pq, Sbus, net, use_load)

        if norm_f < tol:
            if q_ctrl is not None:
                changed, released = q_ctrl.convergence_switch(V)
                if changed:
                    pv, pq, Sbus, net = q_ctrl.pv, q_ctrl.pq, q_ctrl.Sbus, q_ctrl.net
                    Vm = Vm.copy()
                    Vm[released] = q_ctrl.v_set[released]
                    V, f_vec, norm_f = _evaluate(Ybus, Vm, Va, pv, pq, Sbus, net, use_load)
                    continue
            converged = True
            break

    return NRResult(V=V, converged=converged, iterations=iteration, mismatch_max=norm_f)
