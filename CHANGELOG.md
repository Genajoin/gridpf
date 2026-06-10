# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from Conventional Commits via `cz bump`; the git tag is the
authoritative version (`setuptools_scm`).

## [Unreleased]

### Added

- Initial pure-Python Power Flow engine (`gridpf`):
  - Opaque per-unit data contract `PFInput` / `PFOptions` / `PFResult`
    (`gridpf.contract`), versioned (`CONTRACT_VERSION`) with `.npz` serialization.
  - Model-free `solve(net, options, *, init_v=None)`: Gauss-Seidel warm-start +
    Newton-Raphson with optional Q-limit outer loop, DC and soft fallbacks,
    polynomial voltage-dependent load (СХН), warm-start, and controllable-bus
    setpoints (`bus_v_set` / `bus_va_set`) clamped over the start vector.
  - Power-flow numerics adapted from pandapower / PYPOWER (BSD 3-Clause; see
    LICENSE).
