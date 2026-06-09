"""Opaque-контракт данных расчёта установившегося режима (``PFInput`` /
``PFResult``) и опции движка (``PFOptions``).

Самодостаточный, vendor-free контракт: движок ``gridpf`` работает только в
системе p.u. над плоскими numpy-массивами и ничего не знает о конкретном
классе модели сети, формате файла или единицах измерения источника. Граница
данных явная, проверяемая и версионируемая (см. :mod:`gridpf.contract.version`).

Все индексы (``slack_idx``, ``from_idx``, ``to_idx``) — *позиционные* в массиве
шин (``0..n_bus−1``). ``bus_ids`` / ``branch_ids`` — непрозрачные метки строк
(их смысл известен только адаптеру, который собрал ``PFInput`` и пишет результат
обратно по ним); движок ими не пользуется для вычислений.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


BASE_MVA: float = 100.0
"""Базовая мощность пакета (фиксированная, 3-phase basis)."""


# Коды типов шин (позиционно-независимая, собственная конвенция gridpf —
# совпадает с распространённой PF-нотацией: PQ/PV/SLACK).
PQ: int = 0
PV: int = 1
SLACK: int = 2


Method = Literal["gs+nr", "nr", "gs"]


@dataclass
class PFInput:
    """Opaque p.u.-представление сети для расчёта установившегося режима.

    Все индексы (``slack_idx``, ``from_idx``, ``to_idx``) — *позиционные* в
    массиве шин (``0..n_bus−1``); непрозрачные метки строк хранятся в
    ``bus_ids`` / ``branch_ids`` для записи результата обратно адаптером.
    """

    # Топология
    n_bus: int
    n_branch: int
    bus_ids: np.ndarray  # (n_bus,) i8   — непрозрачная метка строки шины
    bus_vn_kv: np.ndarray  # (n_bus,) f8 — базовое напряжение шины, кВ (для write-back)
    bus_type: np.ndarray  # (n_bus,) i1   — 0=PQ, 1=PV, 2=SLACK
    slack_idx: int  # позиционный индекс slack-узла

    branch_ids: np.ndarray  # (n_branch,) i8 — непрозрачная метка строки ветви
    from_idx: np.ndarray  # (n_branch,) i8 — позиционный индекс «от»
    to_idx: np.ndarray  # (n_branch,) i8 — позиционный индекс «до»

    # Параметры ветвей в p.u.
    branch_r: np.ndarray  # последовательное R
    branch_x: np.ndarray  # последовательное X
    branch_g: np.ndarray  # суммарный шунт G ветви (Π-схема)
    branch_b: np.ndarray  # суммарный шунт B ветви (Π-схема)
    branch_g_from: np.ndarray  # шунт со стороны «от»
    branch_b_from: np.ndarray
    branch_g_to: np.ndarray  # шунт со стороны «до»
    branch_b_to: np.ndarray
    tap_ratio: np.ndarray  # модуль прямого Ktr (безразмерное)
    phase_shift: np.ndarray  # arg(Ktr), радианы

    # Шунты узлов (p.u.)
    bus_g_shunt: np.ndarray
    bus_b_shunt: np.ndarray

    # Инъекции для классификации zero-injection (P_gen − P_load, p.u.)
    bus_p_injection: np.ndarray
    bus_q_injection: np.ndarray

    # Q-лимиты генерации в p.u. (NaN → лимит не задан).
    # Используются опционально при enforce_q_lims=True для PV→PQ переключения.
    bus_q_min: np.ndarray | None = None
    bus_q_max: np.ndarray | None = None

    # Базовые компоненты инъекций (для СХН: разделение генерации и нагрузки).
    # ``bus_p_injection = bus_p_gen − bus_p_load`` при a0=1, остальные коэффициенты 0.
    bus_p_gen: np.ndarray | None = None
    bus_q_gen: np.ndarray | None = None
    bus_p_load: np.ndarray | None = None
    bus_q_load: np.ndarray | None = None

    # Коэффициенты СХН на узел (полиномиальная модель степени 2 от |V| в p.u.):
    #   P_load(|V|) = bus_p_load · (a0 + a1·|V| + a2·|V|²)
    #   Q_load(|V|) = bus_q_load · (b0 + b1·|V| + b2·|V|²)
    # По умолчанию a0=1, остальные 0 → константная нагрузка (текущее поведение).
    bus_p_a0: np.ndarray | None = None
    bus_p_a1: np.ndarray | None = None
    bus_p_a2: np.ndarray | None = None
    bus_q_b0: np.ndarray | None = None
    bus_q_b1: np.ndarray | None = None
    bus_q_b2: np.ndarray | None = None

    # Setpoints управляемых узлов (материализованы адаптером, p.u./рад). Клампятся
    # ПОВЕРХ стартового приближения V0 (Rule №4): заданный модуль/угол всегда
    # побеждает тёплый/flat-старт на управляемых узлах.
    #   bus_v_set — заданный |V| (p.u.), обычно на PV/SLACK; NaN → не задан.
    #   bus_va_set — заданный угол (рад), обычно только на SLACK; NaN → не задан.
    bus_v_set: np.ndarray | None = None
    bus_va_set: np.ndarray | None = None

    base_mva: float = BASE_MVA

    @property
    def has_voltage_dependent_load(self) -> bool:
        """Есть ли в сети хотя бы один узел с нетривиальной СХН (a1, a2, b1 или b2 ≠ 0)."""
        for arr in (self.bus_p_a1, self.bus_p_a2, self.bus_q_b1, self.bus_q_b2):
            if arr is not None and np.any(arr != 0.0):
                return True
        return False


@dataclass
class PFOptions:
    """Опции движка расчёта установившегося режима (чистые numeric).

    Не содержит адаптерных забот (write-back, cleanup топологии, источник
    тёплого старта) — они живут в слое-адаптере, вызывающем :func:`gridpf.solve`.

    Attributes:
        method: ``"gs+nr"`` (по умолчанию) — GS warm-start + Newton-Raphson;
            ``"nr"`` — только Newton-Raphson из flat-старта; ``"gs"`` — только
            Gauss-Seidel.
        tol: целевая ∞-норма мощностного небаланса (p.u.).
        max_iter_gs: максимум итераций Gauss-Seidel (в ``"gs+nr"`` — короткий
            warm-start).
        max_iter_nr: максимум итераций Newton-Raphson.
        gs_tol: целевая точность GS-фазы (в ``"gs+nr"`` обычно ``1e-2``).
        enforce_q_lims: переводить PV → PQ при выходе ``Q_calc`` за
            ``[Q_min, Q_max]`` (по умолчанию ``False``, как в ``pandapower.runpp``).
        max_q_lim_swaps: лимит итераций внешнего цикла переключений (защита от
            осцилляций).
        allow_pq_to_pv: разрешить обратный перевод PQ → PV (по умолчанию
            ``False``, MATPOWER-конвенция — резко стабилизирует outer-loop).
        q_lim_top_k: за одну итерацию переключать только ``top_k`` самых тяжёлых
            нарушителей (``None`` — всех сразу).
        dc_fallback: при расхождении NR пробовать DC-warm-start и второй NR-проход.
        use_load_voltage_dependency: учитывать СХН (полиномиальную зависимость
            нагрузки от ``|V|``).
    """

    method: Method = "gs+nr"
    tol: float = 1e-8
    max_iter_gs: int = 10
    max_iter_nr: int = 30
    gs_tol: float = 1e-2
    enforce_q_lims: bool = False
    max_q_lim_swaps: int = 30
    allow_pq_to_pv: bool = False
    q_lim_top_k: int | None = 3
    dc_fallback: bool = True
    use_load_voltage_dependency: bool = True


@dataclass
class PFResult:
    """Результат расчёта установившегося режима.

    Attributes:
        converged: глобальный флаг сходимости (Newton, либо GS если
            ``method="gs"``).
        iterations_gs: число итераций GS-фазы (0 для ``method="nr"``).
        iterations_nr: число итераций NR-фазы (0 для ``method="gs"``).
        V: ``(n_bus,)`` complex — итоговое напряжение в p.u. (в порядке
            ``bus_ids``).
        bus_ids: ``(n_bus,)`` int — соответствующие непрозрачные метки строк шин.
        S_from: ``(n_branch,)`` complex — переток ``Sf = V_f · conj(Yf · V) ·
            S_base`` в МВА (МВт + jМВАр).
        S_to: ``(n_branch,)`` complex — то же со стороны «до».
        mismatch_max: ∞-норма финального небаланса (p.u.).
        method: фактически использованный метод.
    """

    converged: bool
    iterations_gs: int
    iterations_nr: int
    V: np.ndarray
    bus_ids: np.ndarray
    S_from: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    S_to: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    mismatch_max: float = 0.0
    method: str = "gs+nr"
    q_lim_swaps: int = 0
    """Сколько переключений PV ↔ PQ выполнено (только при enforce_q_lims=True)."""
    q_violations: int = 0
    """Сколько PV-узлов в финальном решении нарушают Q-лимиты. Может быть >0
    при ``enforce_q_lims=True`` если outer-loop не сошёлся и сработал
    soft-fallback к чистому NR."""
    failure_reason: str = ""
    """Если ``converged=False`` — короткий код причины: ``max_iter_reached``,
    ``singular_jacobian``, ``no_slack_component``, ``voltage_collapse``,
    ``infeasible_q_lims``. Пустая строка при успехе."""
    voltage_dependent_load_active: bool = False
    """Был ли расчёт выполнен с учётом СХН (``use_load_voltage_dependency=True`` +
    в сети присутствуют узлы с нетривиальной полиномиальной характеристикой)."""

    @property
    def branch_loss_p(self) -> np.ndarray:
        """Активные потери по ветвям, МВт: ``Re(S_from + S_to)``.

        Обе стороны — «в ветвь», поэтому сумма = рассеяние в ветви
        (серия + ветвевые шунты). Главный инженерный критерий качества
        режима — наряду с :attr:`branch_loss_q`.
        """
        return np.asarray(self.S_from.real + self.S_to.real)

    @property
    def branch_loss_q(self) -> np.ndarray:
        """Реактивные потери по ветвям, МВАр: ``Im(S_from + S_to)``.

        Отрицательные значения возможны на слабозагруженных ВЛ —
        зарядка (ветвевые b) превышает потери в X.
        """
        return np.asarray(self.S_from.imag + self.S_to.imag)
