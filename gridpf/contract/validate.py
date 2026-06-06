"""Валидация входного контракта :class:`gridpf.contract.types.PFInput`.

Проверяет структурную согласованность (длины массивов против ``n_bus`` /
``n_branch``, диапазоны индексов) ДО запуска движка. Содержательную проверку
наличия slack-шины и нулевого импеданса ветвей оставляем самому движку
(``classify_buses`` / ``build_ybus`` бросают ``ValueError`` со своими
сообщениями) — чтобы не дублировать и не менять тип исключения.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gridpf.contract.types import PFInput


class PFContractValidationError(ValueError):
    """Вход ``PFInput`` структурно несогласован (strict-режим валидации)."""


@dataclass
class ValidationIssue:
    """Одна проблема валидации входа."""

    field: str
    message: str


@dataclass
class ValidationReport:
    """Итог валидации ``PFInput``."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        if self.issues:
            joined = "; ".join(f"{i.field}: {i.message}" for i in self.issues)
            raise PFContractValidationError(f"PFInput невалиден: {joined}")


_BUS_VECTORS = (
    "bus_ids",
    "bus_vn_kv",
    "bus_type",
    "bus_g_shunt",
    "bus_b_shunt",
    "bus_p_injection",
    "bus_q_injection",
)
_BRANCH_VECTORS = (
    "branch_ids",
    "from_idx",
    "to_idx",
    "branch_r",
    "branch_x",
    "branch_g",
    "branch_b",
    "branch_g_from",
    "branch_b_from",
    "branch_g_to",
    "branch_b_to",
    "tap_ratio",
    "phase_shift",
)


def validate_input(net: PFInput, *, strict: bool = False) -> ValidationReport:
    """Проверить структурную согласованность ``net``.

    Args:
        net: входной контракт.
        strict: если ``True`` — бросить :class:`PFContractValidationError` при
            наличии проблем.

    Returns:
        :class:`ValidationReport` со списком проблем (пуст, если всё согласовано).
    """
    report = ValidationReport()

    for name in _BUS_VECTORS:
        arr = getattr(net, name)
        if arr is None or len(arr) != net.n_bus:
            report.issues.append(
                ValidationIssue(
                    name, f"длина {None if arr is None else len(arr)} != n_bus={net.n_bus}"
                )
            )

    for name in _BRANCH_VECTORS:
        arr = getattr(net, name)
        if arr is None or len(arr) != net.n_branch:
            report.issues.append(
                ValidationIssue(
                    name, f"длина {None if arr is None else len(arr)} != n_branch={net.n_branch}"
                )
            )

    if net.n_branch > 0:
        for name in ("from_idx", "to_idx"):
            idx = np.asarray(getattr(net, name))
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= net.n_bus):
                report.issues.append(
                    ValidationIssue(name, f"индекс вне диапазона [0, {net.n_bus - 1}]")
                )

    if not (0 <= net.slack_idx < net.n_bus):
        report.issues.append(ValidationIssue("slack_idx", f"вне диапазона [0, {net.n_bus - 1}]"))

    if strict:
        report.raise_if_invalid()
    return report
