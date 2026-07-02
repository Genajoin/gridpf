"""Gauss-Seidel итератор для расчёта PF.

Адаптация ``pandapower/pypower/gausspf.py`` (PSERC, BSD) без зависимости от
``ppoption`` и без побочных print-эффектов. Используется как warm-start перед
Newton-Raphson — несколько итераций GS из flat-старта быстро гасят первичные
ошибки модулей напряжений; Newton затем добивает решение до нужной точности.

Алгоритм — классическая последовательная замена для уравнения
``S_calc = V · conj(Ybus · V)``:

* На PQ-шинах: ``V_k ← V_k + (conj(S_k / V_k) − (Ybus_k,: · V)) / Ybus_k,k``.
* На PV-шинах: то же, но Q_k подменяется вычисленным реактивным небалансом
  и затем |V_k| фиксируется на заданном значении.

Slack-шины не обновляются.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridpf.algebra.sbus import compute_sbus
from gridpf.solvers._common import (
    SolverResult,
    mismatch,
    residual_norm,
    residual_vector,
    resolve_use_load,
)


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


GSResult = SolverResult


def gauss_seidel(
    Ybus: csr_matrix,
    Sbus: np.ndarray,
    V0: np.ndarray,
    ref: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    *,
    tol: float = 1e-2,
    max_iter: int = 10,
    network_pu: PFInput | None = None,
    voltage_dependent_load: bool = False,
) -> GSResult:
    """Прогнать Gauss-Seidel до сходимости или ``max_iter``.

    Args:
        Ybus: ``(n_bus, n_bus)`` CSR — узловые проводимости (комплексная p.u.).
        Sbus: ``(n_bus,)`` complex — комплексные инъекции (p.u.). Q PV-шин
            будет переопределяться внутри (но входной массив не модифицируется).
            При ``voltage_dependent_load=True`` пересчитывается каждую итерацию.
        V0: ``(n_bus,)`` complex — стартовое распределение напряжений.
            Обычно flat-старт: |V|=1 для PQ, |V|=Vset для PV/slack, угол=0.
        ref: индексы slack-шин.
        pv: индексы PV-шин.
        pq: индексы PQ-шин.
        tol: целевая ∞-норма небаланса. Для warm-start обычно ``1e-2``…``1e-3``.
        max_iter: максимальное число итераций.
        network_pu: p.u.-представление сети (нужно при ``voltage_dependent_load=True``).
        voltage_dependent_load: учитывать СХН (зависимость нагрузки от ``|V|``).

    Returns:
        ``GSResult`` с финальным напряжением, флагом сходимости, числом
        итераций и достигнутой нормой небаланса.
    """
    if max_iter < 0:
        raise ValueError(f"max_iter должен быть ≥ 0, получено {max_iter}")

    use_load = resolve_use_load(network_pu, voltage_dependent_load)

    V = V0.astype(np.complex128, copy=True)
    Vm = np.abs(V).copy()
    Sbus_local = Sbus.astype(np.complex128, copy=True)
    if use_load:
        assert network_pu is not None
        Sbus_local = compute_sbus(network_pu, V, voltage_dependent=True).copy()

    # ref здесь нужен только для исключения slack из небаланса;
    # PV ∪ PQ — узлы, по активной части которых проверяется сходимость.
    # Начальный небаланс
    mis = mismatch(Ybus, V, Sbus_local)
    f_residual = residual_vector(mis, pv, pq)
    norm_f = residual_norm(f_residual)

    if norm_f < tol or max_iter == 0:
        return GSResult(V=V, converged=norm_f < tol, iterations=0, mismatch_max=norm_f)

    # Для Gauss-Seidel удобнее иметь LIL/CSR с быстрым доступом по строкам.
    # CSR подходит для row slicing, но Ybus[k, k] лучше извлечь заранее.
    diag_y = np.asarray(Ybus.diagonal())

    # Прямой доступ к CSR-массивам: per-узловой row·V через indptr-срез +
    # np.dot (BLAS) вместо Ybus.getrow(k) (строит 1-строчный CSR на каждый
    # узел/итерацию — главный хотспот GS). sort_indices идемпотентен.
    Ybus.sort_indices()
    y_data, y_idx, y_indptr = Ybus.data, Ybus.indices, Ybus.indptr

    iteration = 0
    converged = False
    for iteration in range(1, max_iter + 1):  # noqa: B007 — счётчик нужен в return
        # Обновление PQ-шин. На каждой PQ-шине, если активна СХН, пересчитываем
        # инъекцию через текущий |V_k|; иначе используем константу.
        # СХН диагональна (Sbus[k]=f(|V[k]|)), а V[k] читается ДО своей мутации
        # внизу итерации → пересчёт всех PQ-инъекций один раз перед циклом
        # тождествен пер-узловому (бит-в-бит), но без 1 вызова compute_sbus на узел.
        if use_load:
            assert network_pu is not None
            sv_pre = compute_sbus(network_pu, V, voltage_dependent=True)
            Sbus_local[pq] = sv_pre[pq]
        for k in pq:
            s_k, e_k = y_indptr[k], y_indptr[k + 1]
            row_dot_v = complex(np.dot(y_data[s_k:e_k], V[y_idx[s_k:e_k]]))
            tmp = (np.conj(Sbus_local[k] / V[k]) - row_dot_v) / diag_y[k]
            V[k] = V[k] + tmp

        # Обновление PV-шин: пересчитываем Q, затем угол; |V| фиксируется.
        # На PV-узле Q_load(V_set) = const (т.к. |V| фиксирован), поэтому
        # СХН-поправка к P_inj также константна — её достаточно учесть один раз
        # перед итерацией; здесь нагрузка у PV редка, оставляем как есть.
        if pv.size:
            for k in pv:
                s_k, e_k = y_indptr[k], y_indptr[k + 1]
                row_dot_v = complex(np.dot(y_data[s_k:e_k], V[y_idx[s_k:e_k]]))
                q_calc = (V[k] * np.conj(row_dot_v)).imag
                Sbus_local[k] = Sbus_local[k].real + 1j * q_calc
                tmp = (np.conj(Sbus_local[k] / V[k]) - row_dot_v) / diag_y[k]
                V[k] = V[k] + tmp
            # фиксируем модуль |V| на исходном (заданном) значении
            V[pv] = Vm[pv] * V[pv] / np.abs(V[pv])

        # При активной СХН обновляем PQ-инъекции под новый |V|
        if use_load:
            assert network_pu is not None
            sbus_v = compute_sbus(network_pu, V, voltage_dependent=True)
            Sbus_local[pq] = sbus_v[pq]
            # Real-часть на PV — тоже зависит от V (Q-часть переопределяется ниже)
            for k in pv:
                Sbus_local[k] = sbus_v[k].real + 1j * Sbus_local[k].imag

        mis = mismatch(Ybus, V, Sbus_local)
        f_residual = residual_vector(mis, pv, pq)
        norm_f = residual_norm(f_residual)
        if norm_f < tol:
            converged = True
            break

    return GSResult(V=V, converged=converged, iterations=iteration, mismatch_max=norm_f)
