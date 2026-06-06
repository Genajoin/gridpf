"""Контракт данных расчёта установившегося режима (``PFInput`` / ``PFResult``).

Выделенный, самодостаточный модуль gridpf, владеющий схемой входных/выходных
данных PF и версией контракта. Делает зависимость движка от данных явной,
проверяемой и версионируемой — вместо неявной зависимости от конкретного класса
модели сети.

Состав:

* :mod:`gridpf.contract.types` — ``PFInput`` / ``PFOptions`` / ``PFResult`` +
  константы (``BASE_MVA``, ``PQ``/``PV``/``SLACK``).
* :mod:`gridpf.contract.version` — :data:`CONTRACT_VERSION` + SemVer-логика.
* :mod:`gridpf.contract.validate` — :func:`validate_input`.
* :mod:`gridpf.contract.serialize` — ``.npz``-граница (``save_pf_input`` /
  ``load_pf_input_npz``).
* :mod:`gridpf.contract.runtime` — фасад :func:`solve`.
"""

from gridpf.contract.runtime import solve
from gridpf.contract.serialize import load_pf_input_npz, save_pf_input
from gridpf.contract.types import (
    BASE_MVA,
    PQ,
    PV,
    SLACK,
    Method,
    PFInput,
    PFOptions,
    PFResult,
)
from gridpf.contract.validate import (
    PFContractValidationError,
    ValidationIssue,
    ValidationReport,
    validate_input,
)
from gridpf.contract.version import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_KEY,
    ContractVersion,
    current_version,
    is_data_compatible,
)


__all__ = [
    "BASE_MVA",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_KEY",
    "PQ",
    "PV",
    "SLACK",
    "ContractVersion",
    "Method",
    "PFContractValidationError",
    "PFInput",
    "PFOptions",
    "PFResult",
    "ValidationIssue",
    "ValidationReport",
    "current_version",
    "is_data_compatible",
    "load_pf_input_npz",
    "save_pf_input",
    "solve",
    "validate_input",
]
