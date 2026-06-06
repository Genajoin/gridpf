"""Контракт данных: поля ``PFInput``/``PFResult``, сериализация, версия, валидация."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest

from gridpf import (
    CONTRACT_VERSION,
    PFContractValidationError,
    PFInput,
    PFOptions,
    PFResult,
    is_data_compatible,
    load_pf_input_npz,
    save_pf_input,
    solve,
    validate_input,
)
from gridpf.contract.version import ContractVersion, current_version


_REQUIRED_PFINPUT_FIELDS = {
    "n_bus",
    "n_branch",
    "bus_ids",
    "bus_vn_kv",
    "bus_type",
    "slack_idx",
    "from_idx",
    "to_idx",
    "branch_r",
    "branch_x",
    "bus_p_injection",
    "bus_q_injection",
}
_NEW_SETPOINT_FIELDS = {"bus_v_set", "bus_va_set"}


def test_pfinput_carries_required_and_setpoint_fields() -> None:
    names = {f.name for f in fields(PFInput)}
    assert names >= _REQUIRED_PFINPUT_FIELDS
    assert names >= _NEW_SETPOINT_FIELDS  # материализованные адаптером setpoints


def test_pfresult_fields() -> None:
    names = {f.name for f in fields(PFResult)}
    assert {"converged", "V", "bus_ids", "S_from", "S_to", "mismatch_max"} <= names


def test_serialize_roundtrip_solves_identically(two_bus: PFInput, tmp_path) -> None:
    path = tmp_path / "case.npz"
    save_pf_input(two_bus, path)
    restored = load_pf_input_npz(path)

    # Поля совпадают (включая опущенные None → None).
    for f in fields(PFInput):
        a = getattr(two_bus, f.name)
        b = getattr(restored, f.name)
        if a is None:
            assert b is None, f.name
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=f.name)
        else:
            assert a == b, f.name

    r1 = solve(two_bus, PFOptions())
    r2 = solve(restored, PFOptions())
    np.testing.assert_array_equal(r1.V, r2.V)


def test_optional_none_fields_survive_roundtrip(
    build_net: Callable[..., PFInput], tmp_path
) -> None:
    net = build_net(
        bus_type=[2, 0], edges=[(0, 1, 0.02, 0.06)], p_inj=[0.0, -0.5], q_inj=[0.0, -0.2]
    )
    assert net.bus_v_set is None and net.bus_q_min is None
    path = tmp_path / "n.npz"
    save_pf_input(net, path)
    restored = load_pf_input_npz(path)
    assert restored.bus_v_set is None
    assert restored.bus_q_min is None


def test_contract_version_parses_and_self_compatible() -> None:
    v = ContractVersion.parse(CONTRACT_VERSION)
    assert v == current_version()
    assert is_data_compatible(CONTRACT_VERSION)


def test_incompatible_major_rejected() -> None:
    cur = current_version()
    bumped_major = f"{cur.major + 1}.0.0"
    assert not is_data_compatible(bumped_major)


def test_validate_detects_shape_mismatch(two_bus: PFInput) -> None:
    bad = two_bus
    bad.bus_p_injection = np.zeros(two_bus.n_bus + 1)  # сломанная длина
    report = validate_input(bad, strict=False)
    assert not report.ok
    with pytest.raises(PFContractValidationError):
        validate_input(bad, strict=True)


def test_validate_passes_on_good_input(two_bus: PFInput) -> None:
    assert validate_input(two_bus, strict=False).ok
