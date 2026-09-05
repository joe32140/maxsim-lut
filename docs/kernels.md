# Kernels and dispatch

How the five SIMD kernels differ, how the crate decides which to run, and
which of them has actually executed on real silicon.

## The five kernels

There are five kernels. Which one is fastest is a property of the *core*, not
of the instruction set, so the crate measures instead of guessing.

| kernel | instruction | MACs/instr | query layout |
|---|---|---|---|
| `neon-sdot` | `sdot` | 16 | plane order, 4 rows × 4 tokens in registers |
| `neon-i8mm` | `smmla` | 32 | row pairs, 4 row pairs × 2 token pairs |
| `avx2` | `vpmaddubsw`+`vpmaddwd`+`vpaddd` | 32 per 3 instr | `u8` tiles, 8 rows in lanes |
| `avx2-vnni` | `vpdpbusd` (VEX, 256-bit) | 32 | same tiles as `avx2` |
| `avx512-vnni` | `vpdpbusd` (EVEX, 512-bit) | 64 | same tiles, 16 rows in lanes |

All of them expand a token's packed bytes once through the nibble tables and
reuse those weights across every query row. The x86 kernels and `neon-i8mm`
put query rows into accumulator lanes so no horizontal reduction is ever
needed; `neon-sdot` keeps one accumulator per (row, token) and reduces with a
pairwise tree. `avx512-vnni`, `avx2-vnni` and `neon-i8mm` need no sign
fix-ups, and where the query is stored `+128` as `u8` the exact offset
`128·Σw` is subtracted once per token in integer arithmetic.

**Measured instruction rates** (`isa_peak`, G MAC/s, higher is better):

| core | `sdot` | `smmla` | in the kernel's block shape |
|---|---|---|---|
| Apple M4 | 243 | 277 | sdot 262, smmla 177 |
| Neoverse N2 (`ubuntu-24.04-arm`) | 109 | 217 | sdot 109, smmla 155 |

The M4 issues `smmla` at roughly half `sdot`'s instruction rate, so its
double width nets nothing and the extra interleaving makes it lose; the N2
issues both at the same rate and gains 1.4× in the block shape. Two cores,
same feature bits, opposite answers.

So dispatch **calibrates**: on the first score of a process, each candidate
kernel is checked against the scalar reference on a small synthetic document
and dropped if it disagrees, then the survivors are timed interleaved and the
winner is cached. It costs a few hundred microseconds once. Choosing at
runtime is safe precisely because every kernel is bit-identical, so
calibration can change how long a search takes and never what it returns.

The self-check is there because CI cannot cover every instruction: GitHub's
x86 runners are a mixed pool, and at the time of writing the Linux x86 runner
is an AMD EPYC 7763 with neither AVX-512 nor AVX-VNNI, so `avx2-vnni` is
compile-checked but never executed there. A kernel whose instruction no CI
machine has must prove itself on the host before dispatch will use it; if
none verifies, scoring falls back to the reference path and
`Lut::kernel(dim)` reports `scalar (SelfCheckFailed)`.

Calibration runs inside the first `score` of the process. A server that
would rather not charge one query for it calls `maxsim_lut::warm_up()` at
startup, which makes the decision then and returns the chosen kernel.

Override it when you need to:

```rust
let lut = lut.pin_kernel(Some(maxsim_lut::supported_kernels()[0])); // programmatic
```
```
MAXSIM_LUT_KERNEL=neon-i8mm     # pin by name
MAXSIM_LUT_NO_CALIBRATE=1       # take the first listed kernel, skip measuring
MAXSIM_LUT_FORCE_SCALAR=1       # the reference path
```

Highest precedence first: `MAXSIM_LUT_FORCE_SCALAR`, then the shape checks that
rule SIMD out entirely (`dim % 8`, `MAX_DIM`, no nibble tables), then
`pin_kernel`, then `MAXSIM_LUT_KERNEL`, then calibration (which
`MAXSIM_LUT_NO_CALIBRATE` reduces to "take the widest"). Either pin is ignored
when it names a kernel this CPU cannot execute. `warm_up()` reports the *calibrated*
choice, so it can differ from `Lut::kernel(dim)` when a pin or the forced-scalar
switch is in play.

## What has actually run on silicon

CI can only test the machines it is given, and GitHub's pool changes. As of
the last run:

| kernel | executed on |
|---|---|
| `neon-sdot` | Apple M1, Apple M4, Neoverse N2 |
| `neon-i8mm` | Neoverse N2, Apple M4 |
| `avx2` | AMD EPYC 7763, Intel Xeon |
| `avx512-vnni` | Intel Xeon |
| `avx2-vnni` | **nothing yet** |

`avx2-vnni` is the one path no machine has run, and the reason is structural
rather than bad luck. It needs a CPU with AVX-VNNI but *without* AVX-512-VNNI,
which means Alder Lake or later on the client side, or Zen 5. GitHub's pool
offers AMD Zen 3, which has neither feature, and Intel Xeons that have the
AVX-512 form and not the VEX one, so on every runner so far dispatch has
correctly preferred a different kernel. QEMU is not a way out either: its TCG
backend implements neither extension, so `qemu-x86_64 -cpu max` reports plain
AVX2 and a pinned run would pass without ever executing the kernel.

Two things make shipping it defensible anyway. Its halves are covered
separately: the 8-row tile addressing and masked fold are the AVX2 kernel's,
and `vpdpbusd` with the `128·Σw` correction is the AVX-512 kernel's, both
exercised on real hardware. And the runtime self-check means the untested
combination cannot corrupt a score. On the first CPU that does offer AVX-VNNI,
the kernel is measured against the reference on four shapes before dispatch
will use it, and a disagreement demotes it rather than shipping wrong numbers.
If you would still rather not carry it, pin another kernel with
`Lut::pin_kernel`.

`cargo run --example kernels` prints what the current CPU offers and what
calibration chose; pass kernel names as arguments to make it exit non-zero
when one of them is missing.
