# gridpf

Power system **Power Flow** as a pure-Python library: Newton-Raphson /
Gauss-Seidel / DC on `numpy` + `scipy`, driven by a versioned data
contract.

```
┌──────────────────────── gridpf.PFInput (opaque, p.u.) ───────────────────────┐
│  topology         injections          setpoints           load model (СХН)   │
│  bus_type/idx     P/Q per bus          bus_v_set           polynomial P(V)   │
│  branch r/x/b     gen − load           bus_va_set          Q(V) coefficients │
└───────────────────────────────────────────┬──────────────────────────────────┘
                                            │
                          gridpf.solve(net, options, init_v=…)
                                            │
┌───────────────────────────────────────────▼──────────────────────────────────┐
│  GS warm-start → Newton-Raphson (+ optional Q-lim outer loop) → DC fallback  │
└───────────────────────────────────────────┬──────────────────────────────────┘
                                            │
                gridpf.PFResult: V (complex p.u.), S_from/S_to (MVA),
                iterations, mismatch, convergence, failure reason
```

## Status

Pre-alpha. The public API (`PFInput` / `PFOptions` / `solve` / `PFResult`) and
the data contract version are stable enough to build on; internals may still
move.

## Installation

```bash
pip install gridpf
```

From source (development):

```bash
make setup           # venv + editable install + pre-commit hooks
# or
pip install -e ".[dev,test]"
```

## Quick start

```python
import numpy as np
import gridpf
from gridpf import PFInput, PFOptions, solve

# A 2-bus example: slack(0) feeding a PQ load(1) of 0.5 + j0.2 p.u. over r+jx.
z = 0.02 + 0.06j
net = PFInput(
    n_bus=2, n_branch=1,
    bus_ids=np.array([10, 11]), bus_vn_kv=np.array([110.0, 110.0]),
    bus_type=np.array([2, 0], dtype=np.int8), slack_idx=0,
    branch_ids=np.array([100]),
    from_idx=np.array([0]), to_idx=np.array([1]),
    branch_r=np.array([z.real]), branch_x=np.array([z.imag]),
    branch_g=np.zeros(1), branch_b=np.zeros(1),
    branch_g_from=np.zeros(1), branch_b_from=np.zeros(1),
    branch_g_to=np.zeros(1), branch_b_to=np.zeros(1),
    tap_ratio=np.array([1.0]), phase_shift=np.zeros(1),
    bus_g_shunt=np.zeros(2), bus_b_shunt=np.zeros(2),
    bus_p_injection=np.array([0.0, -0.5]),
    bus_q_injection=np.array([0.0, -0.2]),
)

res = solve(net, PFOptions(method="gs+nr"))
print(res.converged, np.abs(res.V))     # True [1.    0.96...]
```

Warm-start a re-solve from a previous result:

```python
res2 = solve(net, PFOptions(), init_v=res.V)   # fewer NR iterations
```

## Data contract

The input boundary is a versioned `.npz` file — a flat set of per-unit numpy
arrays plus the contract version. Reconstruction needs only numpy, never the
original model, XML, or any vendor code:

```python
from gridpf import save_pf_input, load_pf_input_npz
save_pf_input(net, "case.npz")
net2 = load_pf_input_npz("case.npz")
```

The schema and its SemVer are owned by `gridpf.contract` (`PFInput`,
`CONTRACT_VERSION`, `validate_input`).

## Public API

| Symbol | Purpose |
| --- | --- |
| `solve(net, options=None, *, init_v=None, validate=True)` | run the power flow |
| `PFInput` | opaque per-unit network contract (input) |
| `PFOptions` | engine options (method, tolerances, Q-limits, fallbacks) |
| `PFResult` | result: `V`, `S_from`/`S_to`, iterations, convergence |
| `save_pf_input` / `load_pf_input_npz` | `.npz` contract boundary |
| `validate_input` | structural validation of a `PFInput` |
| `build_ybus(net)` | `(Ybus, Yf, Yt)` assembly (for adapters / branch flows) |
| `CONTRACT_VERSION` | data-contract SemVer |

## Development

```bash
make check        # ruff format + lint + mypy + pytest
make test         # pytest
make type-check   # mypy gridpf
make build        # sdist + wheel
```

## License

MIT — see [LICENSE](LICENSE). The power-flow numerics are adapted from
pandapower / PYPOWER (BSD 3-Clause); the full third-party notice and the list of
adapted files are in the LICENSE file.
