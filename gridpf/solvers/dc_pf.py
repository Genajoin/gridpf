"""DC-приближение PF (linearized power flow) для warm-start.

Стандартное упрощение AC-PF:

* Модули напряжений ``|V| ≡ 1`` p.u.
* Сопротивлением R пренебрегаем (``r << x``).
* sin(δ_i − δ_j) ≈ δ_i − δ_j.

Тогда переток активной мощности по ветви: ``P_ij ≈ (δ_i − δ_j) / x_ij``.
В матричной форме (только активные углы свободных шин):

.. code::

    B' · δ = P

где ``B'`` — матрица susceptance (узловая) без шунтов, без R, без tap-shift.
``P`` — заданные инъекции активной мощности на pv ∪ pq.

Для PF используется как **fallback warm-start**: если NR расходится из
flat-старта, DC-углы дают разумную начальную точку для второго NR-прохода.

См. ``pandapower/pypower/dcpf.py`` (PSERC, BSD) для референса.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def build_b_prime(
    n_bus: int,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    branch_x: np.ndarray,
    tap_ratio: np.ndarray,
) -> csr_matrix:
    """Построить B' для DC-PF.

    Конвенция MATPOWER: ``B'_kk = Σ 1/x_l`` для всех инцидентных ветвей,
    ``B'_kj = -1/x_l`` (если ветвь l между k и j). Tap-ratio влияет: эквивалент
    последовательного импеданса x умножается на ``|t|`` (упрощение
    PSERC dcpf). Шунты и фазоповорот игнорируются.
    """
    # Защита от R≈0 уже в адаптере, но x может быть 0 в специальных случаях.
    x = np.where(branch_x == 0, 1e-9, branch_x) * np.maximum(tap_ratio, 1e-9)
    b = 1.0 / x
    rows = np.concatenate([from_idx, to_idx, from_idx, to_idx])
    cols = np.concatenate([from_idx, to_idx, to_idx, from_idx])
    vals = np.concatenate([b, b, -b, -b])
    B = coo_matrix((vals, (rows, cols)), shape=(n_bus, n_bus), dtype=np.float64).tocsr()
    B.sum_duplicates()
    return B


def dc_powerflow(
    n_bus: int,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    branch_x: np.ndarray,
    tap_ratio: np.ndarray,
    P_inj: np.ndarray,
    ref: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    """Вычислить углы δ (рад) через DC-приближение.

    Args:
        n_bus: число шин.
        from_idx, to_idx: индексы концов ветвей.
        branch_x: реактивные сопротивления ветвей в p.u.
        tap_ratio: модули комплексных коэффициентов трансформации.
        P_inj: ``(n_bus,)`` — активные инъекции (p.u.).
        ref: индексы slack-шин (углы фиксированы = 0).
        pv: индексы PV.
        pq: индексы PQ.

    Returns:
        ``(n_bus,)`` — углы δ в радианах. Slack-углы = 0.
    """
    pvpq = np.concatenate([pv, pq]).astype(np.int64)
    if pvpq.size == 0:
        return np.zeros(n_bus, dtype=np.float64)

    B = build_b_prime(n_bus, from_idx, to_idx, branch_x, tap_ratio)
    # Подматрица B' по pvpq строкам/столбцам.
    B_red = B[pvpq, :][:, pvpq]
    P_red = P_inj[pvpq]

    try:
        delta_red = spsolve(B_red, P_red)
    except RuntimeError:
        # Сингулярная B' — вернуть нулевые углы (худший случай — flat-старт).
        return np.zeros(n_bus, dtype=np.float64)

    delta = np.zeros(n_bus, dtype=np.float64)
    delta[pvpq] = delta_red
    delta[ref] = 0.0
    return delta
