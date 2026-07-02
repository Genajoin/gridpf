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
    * otherwise → the polynomial model evaluated at ``|V|``.
    """
    q_load = net.bus_q_load
    if q_load is None:
        return np.zeros(net.n_bus, dtype=np.float64)
    b0, b1, b2 = net.bus_q_b0, net.bus_q_b1, net.bus_q_b2
    if not voltage_dependent or b0 is None or b1 is None or b2 is None:
        return np.asarray(q_load, dtype=np.float64)
    return _poly_eval(np.asarray(q_load, dtype=np.float64), b0, b1, b2, np.abs(V))


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
    if not voltage_dependent or not net.has_voltage_dependent_load:
        return build_sbus(net)

    Vm = np.abs(V)
    p_load = net.bus_p_load
    q_load = net.bus_q_load
    if p_load is None or q_load is None:
        return build_sbus(net)

    # Полиномиальная СХН: f(|V|) = c0 + c1·|V| + c2·|V|²
    a0, a1, a2 = net.bus_p_a0, net.bus_p_a1, net.bus_p_a2
    b0, b1, b2 = net.bus_q_b0, net.bus_q_b1, net.bus_q_b2
    assert a0 is not None and a1 is not None and a2 is not None
    assert b0 is not None and b1 is not None and b2 is not None

    p_load_v = _poly_eval(p_load, a0, a1, a2, Vm)
    q_load_v = _poly_eval(q_load, b0, b1, b2, Vm)

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
    if not net.has_voltage_dependent_load:
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
