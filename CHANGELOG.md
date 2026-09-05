# Changelog

## 0.1.0 (unreleased)

First release: the asymmetric int8-query × fused-LUT MaxSim kernel from
next-plaid's `residual_lut.rs`, extracted as a dependency-free crate.

- `Lut`, `PreparedQuery`, `Scorer` and `DocView` over borrowed slices, with
  `Packing` as the only host extension point.
- Five SIMD kernels — `neon-sdot`, `neon-i8mm`, `avx2`, `avx2-vnni`,
  `avx512-vnni` — each bit-identical to the scalar reference, plus that
  reference as the fallback for any other target.
- Dispatch calibrates once per process: it verifies every candidate against
  the reference on four shapes, then times the survivors. Which NEON
  instruction wins is a property of the core rather than of the feature bits,
  so measuring beats a feature table. Overridable with `Lut::pin_kernel`,
  `MAXSIM_LUT_KERNEL`, `MAXSIM_LUT_NO_CALIBRATE` and
  `MAXSIM_LUT_FORCE_SCALAR`, and `warm_up()` pays for it at startup.

Measured against the first working version on one machine per core, same day,
alternated: Neoverse N2 2.53×, Apple M1 2.79×, Apple M4 2.16×, AMD EPYC 7763
1.39×. See the README for what each figure comes from.
