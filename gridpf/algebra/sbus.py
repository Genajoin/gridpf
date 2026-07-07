"""Сборка вектора инъекций ``Sbus`` и классификация шин для PF.

``Sbus[i] = (P_gen − P_load) + j(Q_gen − Q_load)`` в p.u. — задание
комплексных инъекций для уравнения мощностного баланса
``S_calc = V · conj(Ybus · V)``.

Конвенция типов шин (``PFInput.bus_type``):

- ``0`` → PQ (нагрузка / контролируемая мощность);
- ``1`` → PV (генератор с заданным |V|);
- ``2`` → SLACK (балансирующий, фиксированный |V|, δ).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridpf.contract.types import PQ, PV, SLACK


if TYPE_CHECKING:
    from gridpf.contract.types import PFInput


def _poly_eval(
    load: np.ndarray,
    c0: np.ndarray | float,
    c1: np.ndarray | float,
    c2: np.ndarray | float,
    Vm: np.ndarray,
) -> np.ndarray:
    """Evaluate the polynomial load model ``load · (c0 + c1·|V| + c2·|V|²)``.

    Single home of the voltage-dependent-load (ZIP-style) formula; every
    consumer (Sbus assembly, Jacobian correction, Q-limit semantics,
    violation reporting) must go through here instead of re-deriving it.
    """
    result: np.ndarray = load * (c0 + c1 * Vm + c2 * Vm * Vm)
    return result


def _critical_effective_vm(
    Vm: np.ndarray,
    v_critical: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rastr U_krit semantics: freeze the load-shape voltage at ``V_crit``.

    Below the per-bus critical voltage the load continues as constant
    impedance: the polynomial (or constant) model is evaluated at ``V_crit``
    and scaled by ``(|V|/V_crit)²``. Returns ``(vm_eff, scale)`` where
    ``scale`` is ``None`` when no bus is below its critical voltage
    (bit-exact legacy path).
    """
    if v_critical is None:
        return Vm, None
    crit = np.asarray(v_critical, dtype=np.float64)
    below = np.isfinite(crit) & (crit > 0.0) & (Vm < crit)
    if not below.any():
        return Vm, None
    vm_eff = np.where(below, crit, Vm)
    safe_crit = np.where(below, crit, 1.0)
    scale = np.where(below, (Vm / safe_crit) ** 2, 1.0)
    return vm_eff, scale


def q_load_at(
    net: PFInput,
    V: np.ndarray,
    *,
    voltage_dependent: bool = True,
) -> np.ndarray:
    """Per-bus reactive load ``Q_load(|V|)`` in p.u.

    Semantics shared by the Q-limit enforcement and the final violation
    report (generator convention: ``Q_gen = Q_inj + Q_load``):

    * ``bus_q_load`` missing → zeros (no load to subtract);
    * ``voltage_dependent=False`` or polynomial coefficients missing →
      the constant ``bus_q_load``;
    * otherwise → the polynomial model evaluated at ``|V|``;
    * with ``net.bus_v_critical`` set (and ``voltage_dependent=True``) —
      constant-impedance continuation below the critical voltage, for both
      polynomial and constant loads.
    """
    q_load = net.bus_q_load
    if q_load is None:
        return np.zeros(net.n_bus, dtype=np.float64)
    q_load = np.asarray(q_load, dtype=np.float64)
    if not voltage_dependent:
        return q_load
    Vm = np.abs(V)
    vm_eff, scale = _critical_effective_vm(Vm, net.bus_v_critical)
    b0, b1, b2 = net.bus_q_b0, net.bus_q_b1, net.bus_q_b2
    if b0 is None or b1 is None or b2 is None:
        q_v = q_load
    else:
        q_v = _poly_eval(q_load, b0, b1, b2, vm_eff)
    if scale is not None:
        q_v = q_v * scale
    return q_v


def build_sbus(net: PFInput) -> np.ndarray:
    """Собрать комплексный вектор инъекций ``Sbus`` (p.u.).

    Не учитывает СХН (нагрузка трактуется как константа). Эквивалент
    ``compute_sbus(net, V=1.0, voltage_dependent=False)``.

    Args:
        net: p.u.-представление сети.

    Returns:
        ``Sbus`` длиной ``n_bus``, dtype=complex128.
    """
    return (net.bus_p_injection + 1j * net.bus_q_injection).astype(np.complex128)


def compute_sbus(
    net: PFInput,
    V: np.ndarray,
    *,
    voltage_dependent: bool = True,
) -> np.ndarray:
    """Собрать ``Sbus(V)`` с учётом полиномиальной СХН на каждом узле.

    .. math::
        P_{load}(|V|) = P_0 \\cdot (a_0 + a_1 \\cdot |V| + a_2 \\cdot |V|^2)

        Q_{load}(|V|) = Q_0 \\cdot (b_0 + b_1 \\cdot |V| + b_2 \\cdot |V|^2)

        S_{inj}(V) = (P_{gen} - P_{load}(|V|)) + j(Q_{gen} - Q_{load}(|V|))

    При ``voltage_dependent=False`` или отсутствии полей СХН в ``net``
    функция возвращает результат, эквивалентный :func:`build_sbus`.

    Args:
        net: p.u.-представление сети.
        V: ``(n_bus,)`` complex — текущее напряжение.
        voltage_dependent: если ``False`` — игнорировать СХН и вернуть константу.

    Returns:
        ``Sbus`` длиной ``n_bus``, dtype=complex128.
    """
    has_poly = net.has_voltage_dependent_load
    has_crit = net.bus_v_critical is not None
    if not voltage_dependent or not (has_poly or has_crit):
        return build_sbus(net)

    Vm = np.abs(V)
    p_load = net.bus_p_load
    q_load = net.bus_q_load
    if p_load is None or q_load is None:
        return build_sbus(net)

    vm_eff, scale = _critical_effective_vm(Vm, net.bus_v_critical)

    # Полиномиальная СХН: f(|V|) = c0 + c1·|V| + c2·|V|² (в точке vm_eff —
    # ниже крита полином заморожен на V_crit, см. _critical_effective_vm).
    a0, a1, a2 = net.bus_p_a0, net.bus_p_a1, net.bus_p_a2
    b0, b1, b2 = net.bus_q_b0, net.bus_q_b1, net.bus_q_b2
    if has_poly:
        assert a0 is not None and a1 is not None and a2 is not None
        assert b0 is not None and b1 is not None and b2 is not None
        p_load_v = _poly_eval(p_load, a0, a1, a2, vm_eff)
        q_load_v = _poly_eval(q_load, b0, b1, b2, vm_eff)
    else:
        # Константная нагрузка (нет СХН-коэффициентов) — сюда попадаем только
        # ради const-Z продолжения ниже крита.
        p_load_v = np.asarray(p_load, dtype=np.float64)
        q_load_v = np.asarray(q_load, dtype=np.float64)
    if scale is not None:
        p_load_v = p_load_v * scale
        q_load_v = q_load_v * scale

    p_gen = net.bus_p_gen
    q_gen = net.bus_q_gen
    assert p_gen is not None and q_gen is not None

    result: np.ndarray = ((p_gen - p_load_v) + 1j * (q_gen - q_load_v)).astype(np.complex128)
    return result


def load_voltage_derivatives(
    net: PFInput,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Производные нагрузки по модулю напряжения для якобиана.

    .. math::
        \\frac{\\partial P_{load}}{\\partial |V|} = P_0 \\cdot (a_1 + 2 a_2 |V|)

        \\frac{\\partial Q_{load}}{\\partial |V|} = Q_0 \\cdot (b_1 + 2 b_2 |V|)

    Returns:
        ``(dP_load_dVm, dQ_load_dVm)`` — массивы длины ``n_bus`` (p.u./p.u.).
        Нули, если СХН отсутствует в сети.
    """
    n = net.n_bus
    has_poly = net.has_voltage_dependent_load
    has_crit = net.bus_v_critical is not None
    if not (has_poly or has_crit):
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    Vm = np.abs(V)
    p_load = net.bus_p_load
    q_load = net.bus_q_load
    a1, a2 = net.bus_p_a1, net.bus_p_a2
    b1, b2 = net.bus_q_b1, net.bus_q_b2

    # d/d|V| of the polynomial model is itself a polynomial with shifted
    # coefficients: load·(c1 + 2·c2·|V|) == _poly_eval(load, c1, 2·c2, 0, |V|).
    if p_load is None or a1 is None or a2 is None:
        dP = np.zeros(n, dtype=np.float64)
    else:
        dP = _poly_eval(p_load, a1, 2.0 * a2, 0.0, Vm)
    if q_load is None or b1 is None or b2 is None:
        dQ = np.zeros(n, dtype=np.float64)
    else:
        dQ = _poly_eval(q_load, b1, 2.0 * b2, 0.0, Vm)

    if has_crit:
        # Ниже крита S_load = S_load(V_crit)·(|V|/V_crit)² →
        # dS/d|V| = S_load(V_crit)·2·|V|/V_crit². Форма нагрузки в точке
        # крита — полином (или константа без коэффициентов).
        crit = np.asarray(net.bus_v_critical, dtype=np.float64)
        below = np.isfinite(crit) & (crit > 0.0) & (Vm < crit)
        if below.any():
            safe_crit = np.where(below, crit, 1.0)
            if p_load is not None:
                a0 = net.bus_p_a0
                p_at_crit = (
                    _poly_eval(np.asarray(p_load, np.float64), a0, a1, a2, safe_crit)
                    if a0 is not None and a1 is not None and a2 is not None
                    else np.asarray(p_load, np.float64)
                )
                dP = np.where(below, p_at_crit * 2.0 * Vm / (safe_crit * safe_crit), dP)
            b0 = net.bus_q_b0
            if q_load is not None:
                q_at_crit = (
                    _poly_eval(np.asarray(q_load, np.float64), b0, b1, b2, safe_crit)
                    if b0 is not None and b1 is not None and b2 is not None
                    else np.asarray(q_load, np.float64)
                )
                dQ = np.where(below, q_at_crit * 2.0 * Vm / (safe_crit * safe_crit), dQ)
    return dP, dQ


def classify_buses(bus_type: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Разбить шины по типам.

    Args:
        bus_type: ``(n_bus,)`` int — коды ``PFInput.bus_type``.

    Returns:
        ``(ref, pv, pq)`` — три массива позиционных индексов:
            * ``ref`` — slack-шины (обычно одна);
            * ``pv``  — PV-шины;
            * ``pq``  — PQ-шины.

    Raises:
        ValueError: если в сети нет slack-шины.
    """
    bus_type = np.asarray(bus_type, dtype=np.int8)
    ref = np.where(bus_type == SLACK)[0].astype(np.int64)
    pv = np.where(bus_type == PV)[0].astype(np.int64)
    pq = np.where(bus_type == PQ)[0].astype(np.int64)
    if ref.size == 0:
        raise ValueError("В сети нет slack-шины (node_type=2). PF требует одну.")
    return ref, pv, pq
