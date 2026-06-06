"""Версия пакета доступна и согласована."""

from __future__ import annotations

import gridpf
from gridpf.contract.version import ContractVersion, current_version


def test_package_version_is_str() -> None:
    assert isinstance(gridpf.__version__, str)
    assert gridpf.__version__  # непустая


def test_contract_version_well_formed() -> None:
    v = current_version()
    assert isinstance(v, ContractVersion)
    assert v.major >= 1
