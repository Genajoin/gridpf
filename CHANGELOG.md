# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from Conventional Commits via `cz bump`; the git tag is the
authoritative version (`setuptools_scm`).

## Unreleased

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
