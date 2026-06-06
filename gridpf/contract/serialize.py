"""Сериализация входного контракта :class:`~gridpf.contract.types.PFInput` в
``.npz``.

Граница данных PF — версионированный ``.npz``-файл: плоские numpy-массивы полей
``PFInput`` + скаляры + версия контракта. Восстановление НЕ требует исходной
модели сети, XML или какого-либо vendor-кода — только numpy. Симметрично
``gridstate.contract.serialize`` для оценки состояния.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from gridpf.contract.types import PFInput
from gridpf.contract.version import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_KEY,
    is_data_compatible,
)


# Поля ``PFInput``, которые не массивы (скаляры) — кладутся отдельным префиксом.
_SCALAR_FIELDS = {"n_bus", "n_branch", "slack_idx", "base_mva"}


def save_pf_input(net: PFInput, path: str | Path) -> None:
    """Сохранить ``net`` в ``.npz`` (сжатый).

    Опциональные поля (``None``) не записываются — на загрузке восстанут как
    ``None`` через дефолты dataclass. Версия контракта пишется отдельным ключом.
    """
    payload: dict[str, np.ndarray] = {CONTRACT_VERSION_KEY: np.array(CONTRACT_VERSION)}
    for f in fields(net):
        val = getattr(net, f.name)
        if val is None:
            continue
        if f.name in _SCALAR_FIELDS:
            payload[f"scalar::{f.name}"] = np.array(val)
        else:
            payload[f"arr::{f.name}"] = np.asarray(val)
    # mypy: **payload статически коллидирует с keyword-only allow_pickle: bool в
    # savez; ключи payload — только имена полей контракта, allow_pickle среди них нет.
    np.savez_compressed(Path(path), **payload)  # type: ignore[arg-type]


def load_pf_input_npz(path: str | Path) -> PFInput:
    """Восстановить :class:`PFInput` из ``.npz``.

    Raises:
        ValueError: если версия контракта в файле несовместима с текущей.
    """
    data = np.load(Path(path), allow_pickle=False)
    if CONTRACT_VERSION_KEY in data.files:
        ver = str(data[CONTRACT_VERSION_KEY])
        if not is_data_compatible(ver):
            raise ValueError(
                f"Версия контракта данных {ver!r} несовместима с текущей {CONTRACT_VERSION!r}."
            )
    kwargs: dict[str, object] = {}
    for f in fields(PFInput):
        scalar_key = f"scalar::{f.name}"
        arr_key = f"arr::{f.name}"
        if scalar_key in data.files:
            raw = data[scalar_key]
            kwargs[f.name] = float(raw) if f.name == "base_mva" else int(raw)
        elif arr_key in data.files:
            kwargs[f.name] = data[arr_key]
    return PFInput(**kwargs)  # type: ignore[arg-type]
