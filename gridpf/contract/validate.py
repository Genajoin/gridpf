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

    _check_q_lock(net, report)
    _check_load_models(net, report)

    if strict:
        report.raise_if_invalid()
    return report


def _check_q_lock(net: PFInput, report: ValidationReport) -> None:
    """``bus_q_lock``: values in {−1, 0, +1}, only on PV buses with that limit set."""
    lock = net.bus_q_lock
    if lock is None:
        return
    lock = np.asarray(lock)
    if lock.shape != (net.n_bus,):
        report.issues.append(ValidationIssue("bus_q_lock", f"shape {lock.shape} != ({net.n_bus},)"))
        return
    if not np.isin(lock, (-1, 0, 1)).all():
        report.issues.append(ValidationIssue("bus_q_lock", "values must be -1, 0 or +1"))
        return
    locked = lock != 0
    if np.any(locked & (np.asarray(net.bus_type) != 1)):
        report.issues.append(ValidationIssue("bus_q_lock", "non-zero lock on a non-PV bus"))
    for side, name, lim in ((1, "bus_q_max", net.bus_q_max), (-1, "bus_q_min", net.bus_q_min)):
        on_side = lock == side
        if not on_side.any():
            continue
        if lim is None or not np.all(np.isfinite(np.asarray(lim)[on_side])):
            report.issues.append(
                ValidationIssue("bus_q_lock", f"lock {side:+d} requires a finite {name}")
            )


_LOAD_MODEL_FIELDS = (
    "load_model_seg_ptr",
    "load_model_seg_v_lo",
    "load_model_seg_coef",
    "bus_p_load_model",
    "bus_q_load_model",
)


def _check_load_models(net: PFInput, report: ValidationReport) -> None:
    """Piecewise load models: consistent CSR table and per-bus model indices."""
    present = [name for name in _LOAD_MODEL_FIELDS if getattr(net, name) is not None]
    if not present:
        return
    if len(present) != len(_LOAD_MODEL_FIELDS):
        missing = sorted(set(_LOAD_MODEL_FIELDS) - set(present))
        report.issues.append(ValidationIssue("load_model_seg_ptr", f"missing fields {missing}"))
        return

    def issue(name: str, message: str) -> None:
        report.issues.append(ValidationIssue(name, message))

    ptr = np.asarray(net.load_model_seg_ptr)
    v_lo = np.asarray(net.load_model_seg_v_lo, dtype=np.float64)
    coef = np.asarray(net.load_model_seg_coef)
    n_seg = v_lo.size
    if ptr.ndim != 1 or ptr.size < 2 or ptr[0] != 0 or ptr[-1] != n_seg or np.any(np.diff(ptr) < 1):
        issue("load_model_seg_ptr", "must start at 0, end at n_seg, every model >= 1 segment")
        return
    if coef.shape != (n_seg, 5):
        issue("load_model_seg_coef", f"shape {coef.shape} != ({n_seg}, 5)")
    if np.isnan(v_lo).any():
        issue("load_model_seg_v_lo", "NaN bound")
        return
    n_model = ptr.size - 1
    for m in range(n_model):
        bounds = v_lo[ptr[m] : ptr[m + 1]]
        if bounds[0] > 0.0:
            issue("load_model_seg_v_lo", f"model {m}: first bound must be <= 0 (normally -inf)")
        if np.any(np.diff(bounds) <= 0.0):
            issue("load_model_seg_v_lo", f"model {m}: bounds must strictly increase")

    for kind, name, load, quad in (
        ("p", "bus_p_load_model", net.bus_p_load, (net.bus_p_a1, net.bus_p_a2)),
        ("q", "bus_q_load_model", net.bus_q_load, (net.bus_q_b1, net.bus_q_b2)),
    ):
        idx = np.asarray(getattr(net, name))
        if idx.shape != (net.n_bus,):
            issue(name, f"shape {idx.shape} != ({net.n_bus},)")
            continue
        if np.any((idx < -1) | (idx >= n_model)):
            issue(name, f"model index outside [-1, {n_model - 1}]")
        on_model = idx >= 0
        if not on_model.any():
            continue
        if load is None:
            issue(name, f"bus_{kind}_load is required for buses with a load model")
        for c in quad:
            if c is not None and np.any(np.asarray(c)[on_model] != 0.0):
                issue(
                    name, f"bus with a {kind.upper()} load model also has a non-trivial quadratic"
                )
                break
