"""gridpf — расчёт установившегося режима (Power Flow).

Чистый движок power-flow на numpy/scipy. Вход — версионированный
opaque-контракт :class:`PFInput` (p.u.), не привязанный к классу модели сети,
формату файла или единицам источника. Публичный API собран здесь; обзор — в
README, детали — в docstring'ах функций.
"""

from importlib.metadata import PackageNotFoundError, version

from gridpf.algebra.ybus import build_ybus
from gridpf.contract import (
    BASE_MVA,
    CONTRACT_VERSION,
    PQ,
    PV,
    SLACK,
    Method,
    PFContractValidationError,
    PFInput,
    PFOptions,
    PFResult,
    ValidationReport,
    is_data_compatible,
    load_pf_input_npz,
    save_pf_input,
    solve,
    validate_input,
)


try:
    # Версия выводится из git-тегов через setuptools_scm и впекается в метаданные
    # пакета при сборке/установке; читаем её обратно из метаданных в рантайме.
    # Фолбэк срабатывает только в «голом» дереве исходников без install/build.
    __version__ = version("gridpf")
except PackageNotFoundError:  # pragma: no cover - un-installed source tree
    __version__ = "0+unknown"


__all__ = [
    "BASE_MVA",
    "CONTRACT_VERSION",
    "PQ",
    "PV",
    "SLACK",
    "Method",
    "PFContractValidationError",
    "PFInput",
    "PFOptions",
    "PFResult",
    "ValidationReport",
    "__version__",
    "build_ybus",
    "is_data_compatible",
    "load_pf_input_npz",
    "save_pf_input",
    "solve",
    "validate_input",
]
