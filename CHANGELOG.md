# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from Conventional Commits via `cz bump`; the git tag is the
authoritative version (`setuptools_scm`).

## v0.6.1 (2026-10-17)

### Fix

- DC warm start anchored at the slack set angles (`bus_va_set`). It solved
  with every slack angle at zero, and Newton keeps the slack fixed, so a
  fallback through it lost the set angles: multi-slack models converged to a
  different regime. Results change only for runs that reach the DC fallback
  with non-zero slack angles; they now match the Newton-only solution.
- Q-limit violations are counted with the deadband widened by the Newton
  reactive tolerance (`tol_q`) when it exceeds the fixed epsilon, so a bus
  pinned at its limit is not reported as a violation because of the reactive
  residual (a correct solution was flagged `infeasible_q_lims` with a loose
  `tol_q`). Unchanged with the default tolerance.

## v0.6.0 (2026-10-17)

### Feat

- Contract 2.1.0 (optional fields only): `PFInput.bus_q_lock` — PV buses
  starting pinned at a Q limit; piecewise-polynomial load models
  (`load_model_seg_ptr` / `load_model_seg_v_lo` / `load_model_seg_coef`,
  `bus_p_load_model` / `bus_q_load_model`), the lower segment applies at a
  boundary.
- `q_lim_mode="inner"`: Q-limit switching inside the Newton iteration with a
  relative PQ→PV deadband and a per-bus return cap.
- Newton step limits and a post-step voltage/angle corridor (`step_*`).
- Separate Newton tolerances `tol_p` / `tol_q`.
- `gs_mode="ordered"` Gauss-Seidel warm start, `gs_guard`,
  `init_v_max_mismatch`, `gs_min_x`.

All new options are off by default (bit-for-bit).

### Fix

- With `enforce_q_lims=True` a failed Q-limit pass is retried with the limits
  from a DC start, and a solution violating the limits is reported as
  `converged=False` (`converged_without_q_lims` / `infeasible_q_lims`)
  instead of silently converged.

## v0.5.0 (2026-10-10)

### BREAKING CHANGE

- Transformer branch shunt model: the whole "from" shunt of a branch
  (`branch_g_from` / `branch_b_from` plus half of `branch_g` / `branch_b`) is
  connected at the "from" node, in the "from"-side base, before the ideal
  transformer. `build_ybus` no longer divides it by `|t|²`
  (`Yff = Ysf + Yc_from`; `Yft`, `Ytf`, `Ytt` are unchanged), so it no longer
  depends on the tap. The contract schema is unchanged, but the meaning of the
  "from" shunt fields on branches with an off-nominal ratio changes, so
  `CONTRACT_VERSION` is bumped to 2.0.0 and `.npz` inputs saved under 1.x are
  rejected by `load_pf_input_npz`; the version policy now treats a change of
  the meaning of an existing field as MAJOR. Inputs that relied on the division (a "from" shunt
  behind the ideal transformer, as in the `makeYbus` form) must divide that
  shunt by `|t|²` themselves. Branches with `|t| = 1` in p.u. and branches
  without a "from" shunt are not affected. Branch flows, `branch_loss_p` /
  `branch_loss_q` and the Jacobian all follow the Y-bus.

## v0.2.0 (2026-06-10)

### Feat

- **contract**: branch_loss_p/branch_loss_q in PFResult (#1)

## v0.1.0 (2026-06-06)

### Feat

- initial gridpf — pure-Python power-flow engine on a versioned p.u. contract
