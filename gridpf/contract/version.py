"""Версия контракта данных PF (``PFInput`` / ``PFResult``).

Контракт данных — публичный API gridpf: формализованный набор полей
``PFInput``/``PFResult`` с зафиксированными формами и dtype-семействами (см.
:mod:`gridpf.contract.types`). Его **версия** — отдельная SemVer-строка,
владеемая здесь и версионируемая вместе с пакетом gridpf.

Политика версий (SemVer для схемы данных):

* **MAJOR** — ломающее изменение схемы: удаление/переименование обязательного
  поля, смена его dtype-семейства. Данные, собранные под более старый major,
  несовместимы.
* **MINOR** — обратносовместимое (аддитивное) изменение: новое необязательное
  поле. Данные более старого minor по-прежнему читаются.
* **PATCH** — изменения, не затрагивающие схему (документация, валидаторы).

Версия едет с сериализованными данными как метаданные
(:data:`CONTRACT_VERSION_KEY`); на входе gridpf проверяет совместимость.
"""

from __future__ import annotations

from dataclasses import dataclass


# Текущая версия контракта данных PF. Стартуем с 1.0.0 — первая явная фиксация
# контракта. Бамп — по политике выше; синхронно обновлять при изменении
# :mod:`gridpf.contract.types`.
CONTRACT_VERSION = "1.0.0"

# Ключ, под которым версия контракта кладётся в метаданные сериализованных
# данных. Валидатор читает его на входе.
CONTRACT_VERSION_KEY = "contract_version"


@dataclass(frozen=True, order=True)
class ContractVersion:
    """Разобранная SemVer-версия контракта (``MAJOR.MINOR.PATCH``).

    Сравнима как кортеж ``(major, minor, patch)`` (``order=True``), поэтому
    пригодна для прямых сравнений. Pre-release/build-метаданные не
    поддерживаются намеренно — контракт версионируется простым SemVer.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> ContractVersion:
        """Разобрать строку ``"MAJOR.MINOR.PATCH"`` в :class:`ContractVersion`.

        Raises:
            ValueError: если формат не ``int.int.int``.
        """
        parts = str(text).strip().split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Версия контракта должна быть 'MAJOR.MINOR.PATCH', получено: {text!r}"
            )
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Нечисловые компоненты версии контракта: {text!r}") from exc
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"Отрицательные компоненты версии контракта: {text!r}")
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, schema: ContractVersion) -> bool:
        """Совместимы ли данные этой версии со схемой версии ``schema``.

        Правило (читатель = ``schema``, данные = ``self``):

        * **major** должны совпадать — иначе схема изменилась несовместимо;
        * **minor** данных не должен превышать minor схемы;
        * **patch** на совместимость не влияет.
        """
        if self.major != schema.major:
            return False
        return self.minor <= schema.minor


def current_version() -> ContractVersion:
    """Разобранная текущая версия контракта (:data:`CONTRACT_VERSION`)."""
    return ContractVersion.parse(CONTRACT_VERSION)


def is_data_compatible(data_version: str) -> bool:
    """Совместима ли версия данных ``data_version`` с текущим контрактом."""
    return ContractVersion.parse(data_version).is_compatible_with(current_version())
