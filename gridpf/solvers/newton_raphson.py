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


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


NRResult = SolverResult


def _sbus_at(
    V: np.ndarray,
    sbus_const: np.ndarray,
    network_pu: PFInput | None,
    voltage_dependent_load: bool,
) -> np.ndarray:
    """Вернуть Sbus с учётом СХН (если задан network_pu и флаг)."""
    if network_pu is None or not voltage_dependent_load:
        return sbus_const
    return compute_sbus(network_pu, V, voltage_dependent=True)


def _try_step(
    Ybus: csr_matrix,
    Vm: np.ndarray,
    Va: np.ndarray,
    dx: np.ndarray,
    mu: float,
    pv: np.ndarray,
    pq: np.ndarray,
    Sbus_const: np.ndarray,
    network_pu: PFInput | None,
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
    Sbus_new = _sbus_at(V_new, Sbus_const, network_pu, voltage_dependent_load)
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
    network_pu: PFInput | None = None,
    voltage_dependent_load: bool = False,
) -> NRResult:
    """Прогнать Newton-Raphson до сходимости или ``max_iter``.

    Args:
        Ybus: ``(n, n)`` CSR.
        Sbus: ``(n,)`` complex — заданные инъекции (p.u.). Используются как
            константные, если ``voltage_dependent_load=False``. При включённой
            СХН служат стартовым приближением, но фактически каждую итерацию
            пересчитываются через ``compute_sbus(network_pu, V)``.
        V0: ``(n,)`` complex — стартовое напряжение (например, после GS warm-start).
        ref: индексы slack-шин.
        pv: индексы PV-шин.
        pq: индексы PQ-шин.
        tol: целевая ∞-норма небаланса.
        max_iter: максимальное число итераций.
        step_control: если ``True`` — backtracking line search по μ при росте
            нормы небаланса (μ ← μ/2, до 5 раз). Расширяет область сходимости
            NR на сетях с большим начальным небалансом. По умолчанию ``True``.
        network_pu: p.u.-представление сети; нужно при ``voltage_dependent_load=True``
            для пересчёта Sbus и поправки якобиана от СХН.
        voltage_dependent_load: учитывать СХН (полиномиальную зависимость
            нагрузки от ``|V|``). По умолчанию ``False`` для backward compat.

    Returns:
        ``NRResult`` с финальным напряжением, флагом сходимости, числом
        итераций и достигнутой нормой небаланса.
    """
    if max_iter < 0:
        raise ValueError(f"max_iter должен быть ≥ 0, получено {max_iter}")
    _ = ref  # slack учитывается через отсутствие в pv/pq; формальный параметр API.

    use_load = resolve_use_load(network_pu, voltage_dependent_load)

    V = V0.astype(np.complex128, copy=True)
    Vm = np.abs(V)
    Va = np.angle(V)

    Sbus_eff = _sbus_at(V, Sbus, network_pu, use_load)
    mis = mismatch(Ybus, V, Sbus_eff)
    f_vec = residual_vector(mis, pv, pq)
    norm_f = residual_norm(f_vec)

    if norm_f < tol or max_iter == 0:
        return NRResult(V=V, converged=norm_f < tol, iterations=0, mismatch_max=norm_f)

    iteration = 0
    converged = False
    for iteration in range(1, max_iter + 1):  # noqa: B007 — счётчик нужен в return
        if use_load:
            assert network_pu is not None
            dP_load, dQ_load = load_voltage_derivatives(network_pu, V)
            dS_load_dVm: np.ndarray | None = (dP_load + 1j * dQ_load).astype(np.complex128)
        else:
            dS_load_dVm = None
        J = build_jacobian(Ybus, V, pv, pq, dS_load_dVm=dS_load_dVm)
        dx = spsolve(J, -f_vec)
        if not np.all(np.isfinite(dx)):
            # Сингулярный якобиан: scipy.spsolve на нём возвращает NaN-вектор
            # (с MatrixRankWarning), а НЕ исключение. Прерываем с последним
            # КОНЕЧНЫМ V; mismatch=nan → _engine классифицирует
            # singular_jacobian и активирует DC-fallback из живой точки
            # (вместо ~30 пустых итераций по NaN).
            norm_f = float("nan")
            break

        # Полный шаг μ=1 пробуем всегда; backtracking активен только если
        # норма выросла И step_control=True.
        Vm_try, Va_try, V_try, f_try, norm_try = _try_step(
            Ybus, Vm, Va, dx, 1.0, pv, pq, Sbus, network_pu, use_load
        )
        if step_control and norm_try > norm_f:
            mu = 0.5
            descent = False
            for _bt in range(5):
                Vm_bt, Va_bt, V_bt, f_bt, norm_bt = _try_step(
                    Ybus, Vm, Va, dx, mu, pv, pq, Sbus, network_pu, use_load
                )
                if norm_bt < norm_try:
                    Vm_try, Va_try, V_try, f_try, norm_try = (Vm_bt, Va_bt, V_bt, f_bt, norm_bt)
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

        Vm, Va, V, f_vec, norm_f = Vm_try, Va_try, V_try, f_try, norm_try

        if norm_f < tol:
            converged = True
            break

    return NRResult(V=V, converged=converged, iterations=iteration, mismatch_max=norm_f)
