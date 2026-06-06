"""ГЛАВНЫЙ ГЕЙТ: ``gridpf.solve`` исполняется только на numpy/scipy.

Движок gridpf не должен тянуть в рантайме никаких внешних vendor-библиотек —
кроме numpy и scipy. В частности, ноль зависимости от ``power_system`` (PSC),
``pandas``, ``pandapower``: контракт ``PFInput`` opaque. Доказательство — прогон
``solve`` на собранном из numpy-массивов ``PFInput`` при заблокированных
«тяжёлых» сторонних зависимостях.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# Внешние библиотеки, которые движок gridpf НЕ должен импортировать в рантайме.
_FORBIDDEN_VENDOR_MODULES = ("pandas", "pandapower", "power_system")


_BUILD_AND_RUN_SRC = textwrap.dedent(
    """
    import numpy as np
    from gridpf import PFInput, PFOptions, solve

    # 3-узловая сеть: slack(0) — PV(1, |V|=1.05) — PQ(2, нагрузка 0.8+j0.4).
    net = PFInput(
        n_bus=3, n_branch=2,
        bus_ids=np.array([10, 11, 12], dtype=np.int64),
        bus_vn_kv=np.array([110.0, 110.0, 110.0]),
        bus_type=np.array([2, 1, 0], dtype=np.int8), slack_idx=0,
        branch_ids=np.array([100, 101], dtype=np.int64),
        from_idx=np.array([0, 1], dtype=np.int64),
        to_idx=np.array([1, 2], dtype=np.int64),
        branch_r=np.array([0.01, 0.02]), branch_x=np.array([0.05, 0.06]),
        branch_g=np.zeros(2), branch_b=np.zeros(2),
        branch_g_from=np.zeros(2), branch_b_from=np.zeros(2),
        branch_g_to=np.zeros(2), branch_b_to=np.zeros(2),
        tap_ratio=np.ones(2), phase_shift=np.zeros(2),
        bus_g_shunt=np.zeros(3), bus_b_shunt=np.zeros(3),
        bus_p_injection=np.array([0.0, 0.3, -0.8]),
        bus_q_injection=np.array([0.0, 0.0, -0.4]),
        bus_v_set=np.array([1.0, 1.05, np.nan]),
    )
    res = solve(net, PFOptions())
    vmax = float(np.max(np.abs(res.V)))
    print(f"RUN_OK converged={bool(res.converged)} it_nr={int(res.iterations_nr)} vmax={vmax:.6f}")
    """
)


def _blocker_src(forbidden: tuple[str, ...]) -> str:
    return textwrap.dedent(
        f"""
        import sys
        import importlib.abc

        _FORBIDDEN = {forbidden!r}


        def _is_forbidden(fullname):
            top = fullname.split(".", 1)[0]
            return top in _FORBIDDEN


        class _VendorBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if _is_forbidden(fullname):
                    raise ImportError("BLOCKED vendor import: " + fullname)
                return None


        for _name in list(sys.modules):
            if _is_forbidden(_name):
                del sys.modules[_name]
        sys.meta_path.insert(0, _VendorBlocker())
        """
    )


_SUBPROCESS_SCRIPT = (
    _blocker_src(_FORBIDDEN_VENDOR_MODULES)
    + textwrap.dedent(
        """
        try:
            import pandas  # noqa: F401

            print("VENDOR_NOT_BLOCKED")
            sys.exit(2)
        except ImportError:
            pass

        import gridpf  # noqa: F401
        """
    )
    + _BUILD_AND_RUN_SRC
)


def test_solve_without_vendor_deps_subprocess() -> None:
    """Главный гейт: ``solve`` исполняется в subprocess с ЗАБЛОКИРОВАННЫМИ vendor-deps."""
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    stdout, stderr = proc.stdout, proc.stderr

    assert "VENDOR_NOT_BLOCKED" not in stdout, (
        f"запрещённая vendor-библиотека оказалась доступна.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert proc.returncode == 0, (
        f"дочерний процесс упал (rc={proc.returncode}).\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "BLOCKED vendor import" not in stderr, (
        f"в трейсе всплыл запрещённый vendor-импорт (рантайм-зависимость не снята):\n{stderr}"
    )

    marker = next((ln for ln in stdout.splitlines() if ln.startswith("RUN_OK")), None)
    assert marker is not None, f"нет маркера RUN_OK в stdout:\n{stdout}"
    assert "converged=True" in marker, f"solve не сошёлся без vendor-deps: {marker}"


def test_subprocess_blocker_actually_blocks() -> None:
    """Контроль негатива: блокатор реально валит ``import pandas`` (rc=2)."""
    script = _blocker_src(_FORBIDDEN_VENDOR_MODULES) + textwrap.dedent(
        """
        try:
            import pandas  # noqa: F401
            print("VENDOR_NOT_BLOCKED")
            sys.exit(0)
        except ImportError:
            print("BLOCK_CONFIRMED")
            sys.exit(2)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2, f"блокатор не сработал: rc={proc.returncode}\n{proc.stdout}"
    assert "BLOCK_CONFIRMED" in proc.stdout
    assert "VENDOR_NOT_BLOCKED" not in proc.stdout
