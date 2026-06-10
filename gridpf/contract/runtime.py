"""Публичная точка входа движка — :func:`solve`.

Тонкий фасад над ядром :func:`gridpf._engine.run_powerflow`: опциональная
валидация входного контракта, затем расчёт.
"""

from __future__ import annotations

import numpy as np

from gridpf._engine import run_powerflow
from gridpf.contract.types import PFInput, PFOptions, PFResult
from gridpf.contract.validate import validate_input


def solve(
    net: PFInput,
    options: PFOptions | None = None,
    *,
    init_v: np.ndarray | None = None,
    validate: bool = True,
) -> PFResult:
    """Рассчитать установившийся режим для opaque-контракта ``net``.

    Args:
        net: входной контракт :class:`~gridpf.contract.types.PFInput` (p.u.).
        options: опции движка; ``None`` → :class:`PFOptions` по умолчанию
            (``"gs+nr"``, СХН включена, dc-fallback включён).
        init_v: ``(n_bus,)`` complex — тёплый старт, выровненный по
            ``net.bus_ids``. ``None`` → flat-старт (бит-в-бит с холодным запуском).
        validate: проверить структурную согласованность входа до расчёта
            (длины массивов, диапазоны индексов). Содержательные ошибки
            (нет slack, нулевой импеданс) движок поднимает сам.

    Returns:
        :class:`~gridpf.contract.types.PFResult`.

    Raises:
        gridpf.contract.validate.PFContractValidationError: при
            ``validate=True`` и структурно несогласованном входе.
        ValueError: если метод неизвестен или в сети нет slack-узла.
        RuntimeError: если якобиан Newton сингулярен.
    """
    if options is None:
        options = PFOptions()
    if validate:
        validate_input(net, strict=True)
    return run_powerflow(net, options, init_v=init_v)
