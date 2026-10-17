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
QLimMode = Literal["outer", "inner"]
GSMode = Literal["classic", "ordered"]


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
    # The "from" shunt (with half of branch_g/branch_b) is connected at the
    # "from" node before the ideal transformer: tap-independent, see algebra.ybus.
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

    # Критическое напряжение нагрузки (p.u.). Ниже
    # ``bus_v_critical[i]`` нагрузка узла продолжается как const-Z:
    #   S_load(|V|) = S_load(V_crit) · (|V| / V_crit)²
    # — полином СХН (или константная нагрузка) вычисляется в точке V_crit и
    # масштабируется квадратично, поэтому фидер, нагруженный «за носом»
    # PV-кривой, остаётся разрешимым (const-Z не создаёт предела передаваемой
    # мощности). NaN у узла → семантика выключена для него. ``None`` (default)
    # — выключено для всей сети, бит-в-бит прежнее поведение. Заполняется
    # адаптером (типовое значение Растра — 0.7·Vnom → 0.7 p.u.).
    bus_v_critical: np.ndarray | None = None

    # Initial Q-limit state of PV buses (int8): −1 — the generator starts
    # pinned at Q_min, +1 — at Q_max (the bus starts as PQ with free |V| and
    # may return to PV under the Q-limit rules), 0 — free. Only honoured with
    # ``PFOptions.enforce_q_lims=True``. ``None`` (default) — all free,
    # bit-for-bit with the previous behavior. Since contract 2.1.0.
    bus_q_lock: np.ndarray | None = None

    # Piecewise-polynomial load models (since contract 2.1.0), shared by buses.
    # Model ``m`` owns segment rows ``load_model_seg_ptr[m]:load_model_seg_ptr[m+1]``
    # (CSR pointer, ``(n_model+1,)`` int64). Each row has a lower voltage bound
    # ``load_model_seg_v_lo`` (p.u. of nominal, strictly increasing within a
    # model, the first one ≤ 0 — normally −inf) and coefficients
    # ``load_model_seg_coef[:, 0..4]`` of ``c0 + c1·v + … + c4·v⁴`` normalised to
    # the nominal load. Segment ``k`` is active for
    # ``v_lo[k] < |V| ≤ v_lo[k+1]``: at a boundary the LOWER segment applies;
    # the value and the Jacobian derivative come from the same segment.
    # ``bus_p_load_model`` / ``bus_q_load_model`` (``(n_bus,)`` int32) pick the
    # model of ``P_load`` / ``Q_load`` per bus; ``−1`` keeps the quadratic
    # ``a``/``b`` coefficients (or the constant load). The critical voltage
    # applies on top, as for the quadratic model. All ``None`` (default) —
    # bit-for-bit with the previous behavior.
    load_model_seg_ptr: np.ndarray | None = None
    load_model_seg_v_lo: np.ndarray | None = None
    load_model_seg_coef: np.ndarray | None = None
    bus_p_load_model: np.ndarray | None = None
    bus_q_load_model: np.ndarray | None = None

    base_mva: float = BASE_MVA

    @property
    def has_poly_load(self) -> bool:
        """Есть ли в сети хотя бы один узел с нетривиальной СХН (a1, a2, b1 или b2 ≠ 0)."""
        for arr in (self.bus_p_a1, self.bus_p_a2, self.bus_q_b1, self.bus_q_b2):
            if arr is not None and np.any(arr != 0.0):
                return True
        return False

    @property
    def has_load_models(self) -> bool:
        """Whether any bus uses a piecewise load model (index ≥ 0)."""
        if self.load_model_seg_ptr is None:
            return False
        for arr in (self.bus_p_load_model, self.bus_q_load_model):
            if arr is not None and np.any(np.asarray(arr) >= 0):
                return True
        return False

    @property
    def has_voltage_dependent_load(self) -> bool:
        """Whether any bus load depends on ``|V|`` (quadratic or piecewise model)."""
        return self.has_poly_load or self.has_load_models


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
        tol_p, tol_q: separate Newton tolerances (p.u.) on the active and the
            reactive residual: converged when ``max|ΔP| < tol_p`` and
            ``max|ΔQ| < tol_q``. ``None`` falls back to ``tol``; both ``None``
            (default) — a single ``tol`` on the whole residual, bit-for-bit.
            They also gate the Q-limit check of ``q_lim_mode="inner"``. The
            Gauss-Seidel phase keeps ``tol`` / ``gs_tol``.
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
        q_lim_tol: deadband на нарушение Q-лимита (p.u.) при проверке PV → PQ:
            узел свопается только если ``Q_gen`` выходит за ``[Q_min − tol,
            Q_max + tol]``. Гасит ложные переключения узлов, чья Q сидит на
            границе и отклоняется на малый residual (например, небаланс мягкого
            IPM при воспроизведении SE-режима). ``0.0`` (default) — прежнее
            поведение бит-в-бит.
        dc_fallback: при расхождении NR пробовать DC-warm-start и второй NR-проход.
        use_load_voltage_dependency: учитывать СХН (полиномиальную зависимость
            нагрузки от ``|V|``).
        v_plausible_range: ``(lo, hi)`` — диапазон физического правдоподобия
            ``|V|`` p.u. финального решения. NR может численно сойтись
            (mismatch < tol) в физически бессмысленную точку нижней ветви
            PV-кривой (|V| ~ 0.1–0.3 p.u. при несогласованных инжекциях) —
            такой режим помечается ``converged=False`` с
            ``failure_reason="implausible_voltage"``. Диапазон намеренно
            экстремальный (легитимные тяжёлые режимы не задевает);
            ``None`` — проверка выключена (прежнее поведение).
        step_max_dvm: Newton step scaling on the voltage modulus — the whole
            increment is scaled so that ``|Δ|V|| / |V|²`` (relative step
            divided by ``|V|`` in p.u.) stays within the limit. ``None`` — off.
        step_max_dva: Newton step scaling on the per-bus angle step, rad.
        step_max_branch_dva: Newton step scaling on the per-branch
            angle-difference step ``|Δδ_i − Δδ_j|``, rad.
        step_v_corridor: ``(v_lo, v_hi)`` p.u. — post-step corridor: a bus
            crossing its boundary rolls the whole step back onto it
            (from the second Newton iteration on).
        step_max_branch_angle: post-step corridor on ``|δ_i − δ_j|`` over
            branches, rad.

        q_lim_mode: where Q limits are checked (with ``enforce_q_lims=True``).
            ``"outer"`` (default) — outer loop after a fully converged Newton
            pass, ``q_lim_top_k`` worst violators per pass, PQ→PV only with
            ``allow_pq_to_pv``. ``"inner"`` — inside the Newton iteration:
            whenever the residual is within ``tol`` one direction group
            switches at a fixed priority (PV→Qmax with Qmin→PV first, then
            PV→Qmin with Qmax→PV), all its candidates at once; pinned buses
            drifting from ``V_set`` in the wrong direction are also returned
            to PV after ordinary steps. ``q_lim_top_k`` and ``allow_pq_to_pv``
            are ignored in this mode; ``q_lim_tol`` is the Q deadband.
        q_lim_v_deadband: ``"inner"`` mode — relative ``|V|`` deadband of the
            PQ→PV return at a converged point: a bus pinned at Qmax returns
            when ``(|V| − V_set)/V_set > deadband`` (symmetrically at Qmin).
        q_lim_max_returns: ``"inner"`` mode — returns to PV allowed per bus
            and direction; after that the bus stays pinned.

        gs_mode: Gauss-Seidel warm start. ``"classic"`` (default) — buses in
            index order, constant-power load, up to ``max_iter_gs`` iterations
            down to ``gs_tol``. ``"ordered"`` — exactly ``max_iter_gs`` passes
            in the order of the referred series impedance from the slack,
            loads with ``P > 0, Q > 0`` taken as admittances at the current
            ``|V|`` (see :mod:`gridpf.solvers.gauss_seidel_ordered`).
        gs_min_x: ``"ordered"`` mode — scale of the small inductive reactance
            that replaces a series-capacitive branch (``x < 0``) during
            Gauss-Seidel: ``x_eq = 55·gs_min_x·U_from`` Ω, ``U_from`` — nominal
            voltage of the branch "from" bus in kV (``4e-6`` → ``2.2e-4·U`` Ω).
        gs_guard: discard the Gauss-Seidel result and start Newton from the
            pre-GS voltage when GS made the residual worse, produced a
            non-finite value or left ``step_v_corridor``. ``False`` — off.
        init_v_max_mismatch: with a warm start ``init_v``, fall back to the
            flat start when the residual at ``init_v`` exceeds this value
            (p.u.). ``None`` — off.

        All ``step_*`` limits default to ``None`` (bit-for-bit with the
        unlimited step); see :mod:`gridpf.solvers.step_limits`.
    """

    method: Method = "gs+nr"
    tol: float = 1e-8
    tol_p: float | None = None
    tol_q: float | None = None
    max_iter_gs: int = 10
    max_iter_nr: int = 30
    gs_tol: float = 1e-2
    enforce_q_lims: bool = False
    max_q_lim_swaps: int = 30
    allow_pq_to_pv: bool = False
    q_lim_top_k: int | None = 3
    q_lim_tol: float = 0.0
    dc_fallback: bool = True
    use_load_voltage_dependency: bool = True
    v_plausible_range: tuple[float, float] | None = (0.5, 1.5)
    step_max_dvm: float | None = None
    step_max_dva: float | None = None
    step_max_branch_dva: float | None = None
    step_v_corridor: tuple[float, float] | None = None
    step_max_branch_angle: float | None = None
    q_lim_mode: QLimMode = "outer"
    q_lim_v_deadband: float = 1e-5
    q_lim_max_returns: int = 4
    gs_mode: GSMode = "classic"
    gs_min_x: float = 4e-6
    gs_guard: bool = False
    init_v_max_mismatch: float | None = None


@dataclass
class PFResult:
    """Результат расчёта установившегося режима."""

    converged: bool
    """глобальный флаг сходимости (Newton, либо GS если ``method="gs"``)."""
    iterations_gs: int
    """число итераций GS-фазы (0 для ``method="nr"``)."""
    iterations_nr: int
    """число итераций NR-фазы (0 для ``method="gs"``)."""
    V: np.ndarray
    """``(n_bus,)`` complex — итоговое напряжение в p.u. (в порядке ``bus_ids``)."""
    bus_ids: np.ndarray
    """``(n_bus,)`` int — соответствующие непрозрачные метки строк шин."""
    S_from: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    """``(n_branch,)`` complex — переток ``Sf = V_f · conj(Yf · V) · S_base`` в
    МВА (МВт + jМВАр)."""
    S_to: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    """``(n_branch,)`` complex — то же со стороны «до»."""
    mismatch_max: float = 0.0
    """∞-норма финального небаланса (p.u.)."""
    method: str = "gs+nr"
    """фактически использованный метод."""
    q_lim_swaps: int = 0
    """Сколько переключений PV ↔ PQ выполнено (только при enforce_q_lims=True)."""
    q_violations: int = 0
    """Сколько PV-узлов в финальном решении нарушают Q-лимиты. Может быть >0
    при ``enforce_q_lims=True`` если outer-loop не сошёлся и сработал
    soft-fallback к чистому NR."""
    failure_reason: str = ""
    """Если ``converged=False`` — короткий код причины: ``max_iter_reached``,
    ``singular_jacobian``, ``no_slack_component``, ``voltage_collapse``,
    ``infeasible_q_lims``, ``implausible_voltage``, ``converged_without_q_lims``.
    Пустая строка при успехе. With ``enforce_q_lims=True`` a solution that
    violates the generator Q limits is never ``converged``: ``infeasible_q_lims``
    when the Q-limit pass ended with violations, ``converged_without_q_lims``
    when only the fallback without Q limits solved it (``V`` is that solution,
    ``q_violations`` counts its violations)."""
    implausible_v_nodes: int = 0
    """Узлов с ``|V|`` вне ``v_plausible_range`` в численно сошедшемся решении
    (>0 ⇒ ``converged=False``, ``failure_reason="implausible_voltage"``)."""
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
