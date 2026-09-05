# Testing

The contract this crate has to hold is that every SIMD path returns the same
bits as the scalar reference, because that is what makes choosing a kernel at
runtime safe. The tests below are aimed at that one property.

```
cargo test
MAXSIM_LUT_FORCE_SCALAR=1 cargo test
```

Two tests pin the bit-exactness contract across every kernel the CPU can
execute, not just the one dispatch picks: the unit test
`every_supported_kernel_matches_scalar_bitwise` over a shape sweep (`nq` 1–32
× `nbits` 1/2/4 × `dim` 8–256 × with and without each optional term, three
independent random draws each), and the integration test
`pinned_kernels_agree_with_the_calibrated_default` through the public pinning
API. So an AVX-512 machine still verifies AVX2, and a Neoverse core verifies
both NEON kernels.

`randomised_shapes_match_the_scalar_reference` then draws 250 shapes at
random rather than from a grid, varying code width, dimension, query rows,
token count and row padding together, and asserts that the mix actually
reached both a SIMD kernel and the scalar fallback. A hand-picked sweep
covers the boundaries someone thought of, which is the set a bug in the
tails is least likely to occupy.

`tests/concurrency.rs` is a separate binary, so its threads race for the
first dispatch in a cold process: it checks that concurrent scoring is
bit-identical to a single-threaded run, that every thread resolves
calibration to the same kernel, and that the public types are `Send + Sync`.
`the_self_check_rejects_a_kernel_that_disagrees` feeds calibration a runner
that lies about one kernel and asserts that exactly that kernel is dropped,
because an untested safety net is not a safety net.

CI runs x86_64 and aarch64 runners, a `cargo check` for a target with no SIMD
path, and the MSRV on both architectures.
