"""Якобиан мощностного небаланса для Newton-Raphson PF.

Формулы — стандартные pandapower/MATPOWER (см. ``dSbus_dV.py`` PSERC):

.. code::

    Ibus       = Ybus · V
    diagV      = diag(V)
    diagIbus   = diag(Ibus)
    diagVnorm  = diag(V / |V|)

    dS/d|V|    = diagV · conj(Ybus · diagVnorm) + conj(diagIbus) · diagVnorm
    dS/dδ      = j · diagV · conj(diagIbus − Ybus · diagV)

Подматрицы для Newton:

.. code::

    H = ∂P/∂δ  = Re(dS/dδ)
    N = ∂P/∂|V|= Re(dS/d|V|)
    J = ∂Q/∂δ  = Im(dS/dδ)
    L = ∂Q/∂|V|= Im(dS/d|V|)

Активная часть якобиана собирается с строчно-столбцовым выбором по
``pvpq = pv ∪ pq`` для P-уравнений и ``pq`` для Q-уравнений.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


def dSbus_dV(Ybus: csr_matrix, V: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    """Частные производные ``S = V · conj(Ybus · V)`` по ``|V|`` и ``δ``.

    Args:
        Ybus: ``(n, n)`` CSR — узловые проводимости.
        V: ``(n,)`` complex — текущие комплексные напряжения.

    Returns:
        ``(dS/d|V|, dS/dδ)`` — обе CSR-матрицы ``(n, n)``.
    """
    n = V.size
    ib = np.arange(n, dtype=np.int64)
    Ibus = Ybus @ V

    diagV = csr_matrix((V, (ib, ib)), shape=(n, n))
    diagIbus = csr_matrix((Ibus, (ib, ib)), shape=(n, n))
    Vnorm = V / np.abs(V)
    diagVnorm = csr_matrix((Vnorm, (ib, ib)), shape=(n, n))

    dS_dVm = diagV @ (Ybus @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
    dS_dVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conjugate()

    return dS_dVm.tocsr(), dS_dVa.tocsr()


def build_jacobian(
    Ybus: csr_matrix,
    V: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    *,
    dS_load_dVm: np.ndarray | None = None,
) -> csr_matrix:
    """Собрать активный якобиан размера ``(n_pvpq + n_pq, n_pvpq + n_pq)``.

    Структура:

    .. code::

        [ H   N ]
        [ J   L ]

    где::

        H = (dP/dδ)[pvpq, pvpq]
        N = (dP/d|V|)[pvpq, pq]
        J = (dQ/dδ)[pq, pvpq]
        L = (dQ/d|V|)[pq, pq]

    Args:
        Ybus: ``(n, n)`` CSR.
        V: ``(n,)`` complex — текущая итерация напряжений.
        pv: индексы PV-шин.
        pq: индексы PQ-шин.
        dS_load_dVm: ``(n,)`` complex или ``None``. Поправка от СХН на
            диагонали ``dS/d|V|``: ``∂P_load/∂|V| + j·∂Q_load/∂|V|``.
            Прибавляется к ``dS_dVm`` перед извлечением блоков.
            Знак ``+`` — потому что в residual'е ``F = S_calc − S_inj``,
            а ``S_inj = S_gen − S_load``, и ``∂F/∂|V|`` получает ``+∂S_load/∂|V|``.

    Returns:
        Якобиан CSR.
    """
    pvpq = np.concatenate([pv, pq]).astype(np.int64)
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

    if dS_load_dVm is not None:
        n = V.size
        ib = np.arange(n, dtype=np.int64)
        diag_load = csr_matrix((dS_load_dVm.astype(np.complex128), (ib, ib)), shape=(n, n))
        dS_dVm = dS_dVm + diag_load

    # Извлекаем нужные блоки. Slicing CSR по строкам и столбцам — поддерживается
    # scipy: M[rows, :][:, cols].
    H = dS_dVa[pvpq, :][:, pvpq].real
    N = dS_dVm[pvpq, :][:, pq].real
    J = dS_dVa[pq, :][:, pvpq].imag
    L = dS_dVm[pq, :][:, pq].imag

    top = hstack([H, N], format="csr")
    bot = hstack([J, L], format="csr")
    return vstack([top, bot], format="csr")
