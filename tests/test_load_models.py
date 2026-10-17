"""Tests for piecewise-polynomial load models (``load_model_seg_*``)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gridpf import PFInput, PFOptions, solve
from gridpf.algebra.sbus import compute_sbus, load_voltage_derivatives, q_load_at
from gridpf.contract.serialize import load_pf_input_npz, save_pf_input
from gridpf.contract.validate import PFContractValidationError, validate_input
from tests.test_load_voltage import _build_three_bus


INF = np.inf
# Model 0: P curve; model 1/2: the two standard piecewise Q curves.
SEG_PTR = np.array([0, 1, 4, 7], dtype=np.int64)
SEG_V_LO = np.array([-INF, -INF, 0.815, 1.2, -INF, 0.815, 1.2])
SEG_COEF = np.array(
    [
        [0.83, -0.3, 0.47, 0.0, 0.0],
        [0.721, 0.15971, 0.0, 0.0, 0.0],
        [3.7, -7.0, 4.3, 0.0, 0.0],
        [1.492, 0.0, 0.0, 0.0, 0.0],
        [0.657, 0.159135, 0.0, 0.0, 0.0],
        [4.9, -10.1, 6.2, 0.0, 0.0],
        [1.708, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _with_models(net: PFInput, p_model: int = 0, q_model: int = 1) -> PFInput:
    return replace(
        net,
        load_model_seg_ptr=SEG_PTR,
        load_model_seg_v_lo=SEG_V_LO,
        load_model_seg_coef=SEG_COEF,
        bus_p_load_model=np.array([-1, -1, p_model], dtype=np.int32),
        bus_q_load_model=np.array([-1, -1, q_model], dtype=np.int32),
    )


def _v(vm2: float) -> np.ndarray:
    return np.array([1.0, 1.0, vm2], dtype=np.complex128)


class TestEvaluation:
    @pytest.mark.parametrize(
        ("vm", "shape"),
        [
            (0.5, 0.721 + 0.15971 * 0.5),  # lower linear segment
            (0.815, 0.721 + 0.15971 * 0.815),  # boundary: LOWER segment applies
            (1.0, 1.0),
            (1.2, 3.7 - 7.0 * 1.2 + 4.3 * 1.44),  # boundary: lower (parabola)
            (1.3, 1.492),
        ],
    )
    def test_q_curve_segments(self, vm: float, shape: float) -> None:
        net = _with_models(_build_three_bus())
        q_load = q_load_at(net, _v(vm))
        assert q_load[2] == pytest.approx(0.20 * shape, abs=1e-12)
        assert q_load[1] == pytest.approx(0.10)  # bus without a model: constant

    def test_sbus_uses_p_and_q_models(self) -> None:
        net = _with_models(_build_three_bus())
        s = compute_sbus(net, _v(0.7))
        assert -s[2].real == pytest.approx(0.50 * (0.83 - 0.3 * 0.7 + 0.47 * 0.49), abs=1e-12)
        assert -s[2].imag == pytest.approx(0.20 * (0.721 + 0.15971 * 0.7), abs=1e-12)

    def test_derivative_from_the_same_segment(self) -> None:
        net = _with_models(_build_three_bus())
        _, dq_low = load_voltage_derivatives(net, _v(0.815))  # boundary → lower segment
        assert dq_low[2] == pytest.approx(0.20 * 0.15971, abs=1e-12)
        _, dq_mid = load_voltage_derivatives(net, _v(1.0))
        assert dq_mid[2] == pytest.approx(0.20 * (-7.0 + 2 * 4.3), abs=1e-12)
        _, dq_top = load_voltage_derivatives(net, _v(1.3))
        assert dq_top[2] == 0.0

    @pytest.mark.parametrize("vm", [0.6, 0.9, 1.1, 1.25])
    def test_derivative_matches_finite_difference(self, vm: float) -> None:
        net = _with_models(_build_three_bus(), q_model=2)
        h = 1e-7
        dp, dq = load_voltage_derivatives(net, _v(vm))
        s_hi = compute_sbus(net, _v(vm + h))
        s_lo = compute_sbus(net, _v(vm - h))
        fd = -(s_hi - s_lo) / (2 * h)
        assert dp[2] == pytest.approx(fd[2].real, rel=1e-6)
        assert dq[2] == pytest.approx(fd[2].imag, rel=1e-6)

    def test_critical_voltage_on_top_of_model(self) -> None:
        net = replace(
            _with_models(_build_three_bus()), bus_v_critical=np.array([np.nan, np.nan, 0.7])
        )
        q_load = q_load_at(net, _v(0.5))
        at_crit = 0.721 + 0.15971 * 0.7
        assert q_load[2] == pytest.approx(0.20 * at_crit * (0.5 / 0.7) ** 2, abs=1e-12)


class TestBitForBit:
    def test_quadratic_migrated_to_one_segment_model(self) -> None:
        # Former quadratic a/b coefficients moved into single-segment models.
        legacy = _build_three_bus(p_a=(0.83, -0.3, 0.47), q_b=(3.7, -7.0, 4.3))
        migrated = replace(
            _build_three_bus(),
            load_model_seg_ptr=np.array([0, 1, 2], dtype=np.int64),
            load_model_seg_v_lo=np.array([-INF, -INF]),
            load_model_seg_coef=np.array(
                [[0.83, -0.3, 0.47, 0.0, 0.0], [3.7, -7.0, 4.3, 0.0, 0.0]]
            ),
            bus_p_load_model=np.array([-1, -1, 0], dtype=np.int32),
            bus_q_load_model=np.array([-1, -1, 1], dtype=np.int32),
        )
        for vm in (0.6, 0.93, 1.07):
            np.testing.assert_array_equal(
                compute_sbus(migrated, _v(vm)), compute_sbus(legacy, _v(vm))
            )
            np.testing.assert_array_equal(
                load_voltage_derivatives(migrated, _v(vm))[1],
                load_voltage_derivatives(legacy, _v(vm))[1],
            )
        np.testing.assert_array_equal(solve(migrated).V, solve(legacy).V)

    def test_all_minus_one_is_legacy(self) -> None:
        legacy = _build_three_bus(q_b=(0.0, 0.0, 1.0))
        net = replace(_with_models(legacy), bus_p_load_model=np.full(3, -1, np.int32))
        net = replace(net, bus_q_load_model=np.full(3, -1, np.int32))
        np.testing.assert_array_equal(solve(net).V, solve(legacy).V)


class TestSolve:
    def test_heavy_load_differs_from_single_parabola(self) -> None:
        # Load heavy enough to push |V| below the 0.815 break of the Q curve.
        base = _build_three_bus()
        heavy = replace(
            base,
            bus_p_load=np.array([0.0, 0.3, 1.6]),
            bus_q_load=np.array([0.0, 0.1, 0.6]),
            bus_p_injection=np.array([0.0, -0.3, -1.6]),
            bus_q_injection=np.array([0.0, -0.1, -0.6]),
        )
        piecewise = solve(_with_models(heavy), PFOptions(method="nr"))
        parabola = solve(
            replace(
                heavy,
                bus_q_b0=np.array([1.0, 1.0, 3.7]),
                bus_q_b1=np.array([0.0, 0.0, -7.0]),
                bus_q_b2=np.array([0.0, 0.0, 4.3]),
                bus_p_a0=np.array([1.0, 1.0, 0.83]),
                bus_p_a1=np.array([0.0, 0.0, -0.3]),
                bus_p_a2=np.array([0.0, 0.0, 0.47]),
            ),
            PFOptions(method="nr"),
        )
        assert piecewise.converged and parabola.converged
        assert abs(piecewise.V[2]) < 0.815
        assert abs(abs(piecewise.V[2]) - abs(parabola.V[2])) > 1e-3


class TestContract:
    def test_valid(self) -> None:
        assert validate_input(_with_models(_build_three_bus())).ok

    def test_npz_round_trip(self, tmp_path: object) -> None:
        net = _with_models(_build_three_bus())
        path = f"{tmp_path}/net.npz"
        save_pf_input(net, path)
        back = load_pf_input_npz(path)
        np.testing.assert_array_equal(back.load_model_seg_coef, SEG_COEF)
        np.testing.assert_array_equal(back.bus_q_load_model, net.bus_q_load_model)

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            ({"load_model_seg_ptr": np.array([0, 1, 4, 6])}, "load_model_seg_ptr"),
            ({"load_model_seg_v_lo": SEG_V_LO[[0, 1, 3, 2, 4, 5, 6]]}, "load_model_seg_v_lo"),
            ({"load_model_seg_v_lo": np.r_[0.1, SEG_V_LO[1:]]}, "load_model_seg_v_lo"),
            ({"load_model_seg_coef": SEG_COEF[:, :3]}, "load_model_seg_coef"),
            ({"bus_q_load_model": np.array([-1, -1, 3], np.int32)}, "bus_q_load_model"),
            ({"bus_q_load_model": None}, "load_model_seg_ptr"),
        ],
    )
    def test_invalid(self, change: dict[str, object], field: str) -> None:
        net = replace(_with_models(_build_three_bus()), **change)  # type: ignore[arg-type]
        with pytest.raises(PFContractValidationError, match=field):
            validate_input(net, strict=True)

    def test_model_and_quadratic_on_same_bus_rejected(self) -> None:
        net = _with_models(_build_three_bus(q_b=(0.0, 0.0, 1.0)))
        with pytest.raises(PFContractValidationError, match="non-trivial quadratic"):
            validate_input(net, strict=True)
