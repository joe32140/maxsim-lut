# Benchmarks

Every figure here was measured with the arms interleaved in one process and
reduced by minimum, on a machine whose load is reported alongside the result.
Cross-process comparison on a hybrid CPU can invert a result outright.

## Kernel against the scalar reference

The `Nx scalar` column below is against this crate's *own* scalar reference,
which already uses the fused table — it is a kernel-vs-kernel figure for
tuning, not the host-visible speedup. For that, see [inside a real
engine](#inside-a-real-engine) below, or the summary in the README.

`cargo run --release --example bench` times **every kernel this CPU can run**,
plus the scalar reference, interleaved round by round in one process, and
asserts all of them produce the same checksum:

```text
arch aarch64, dispatched kernel: neon-sdot, 9 interleaved rounds
                 neon-i8mm:   31.10 ns/token  (median   33.59,   7.5 µs/doc,  5.63x scalar)
    neon-sdot (dispatched):   29.63 ns/token  (median   31.84,   7.1 µs/doc,  5.91x scalar)
          scalar reference:  175.07 ns/token  (median  176.70,  42.0 µs/doc,  1.00x scalar)
                     noise: 8.0% median-vs-best spread — quiet enough to compare kernels
```

Comparing separate `cargo run` invocations instead is how a busy machine
invents a result: on this hybrid M4 two single-shot runs put `neon-i8mm`
40% *ahead* of `neon-sdot`, because one of them landed on an efficiency core.
Read the noise line before believing a difference. On Apple Silicon with an
x86 rustup default, pass `--target aarch64-apple-darwin`.

`cargo run --release --example isa_peak` reports the sustained rate of each
int8 dot instruction on the current core, which is what the per-core kernel
choice rests on — see [kernels.md](kernels.md).

## How much faster than the first version

Three structural changes separate the current kernels from the crate's first
commit (`aa2ecba`, the straight extraction from next-plaid): register
blocking, the GEMM-tile form on x86, and the per-core kernel choice described
in [kernels.md](kernels.md).
The `v1-comparison` workflow builds that commit beside current `main` and
alternates the two on one machine, so each row below is a single same-day
measurement rather than a figure stitched across harnesses (32 query rows ×
240 tokens, dim 128, nbits 4, ns per scored token, minimum of 5 alternated
rounds, bit-identical results throughout):

| core | v1 | now | total | where it comes from |
|---|---|---|---|---|
| Neoverse N2 (`ubuntu-24.04-arm`) | 159.7 | 63.2 | **2.53×** | blocking 2.16×, then `smmla` 1.17× |
| Apple M1 (`macos-14`) | 120.9 | 43.3 | **2.79×** | blocking only; the core has no I8MM |
| Apple M4 | 63.8 | 29.5 | **2.16×** | blocking only; `smmla` measured slower here |
| AMD EPYC 7763 (`ubuntu-latest`) | 146.6 | 105.7 | **1.39×** | GEMM tiles only; Zen 3 has no VNNI |

The AVX-512 column is missing because the machine that produced it is gone:
GitHub's x86 pool served an Intel part with AVX-512 VNNI in the morning and
an AMD EPYC 7763 by the afternoon. On that Intel runner the same progression
measured 143.8 → 86.3 (blocking) → **37.2** (tiles), a 3.87× total, and
`avx2-vnni` did not exist yet. Treat those three numbers as historical.

Two results worth keeping in mind when reading the table:

* **The same change is worth wildly different amounts per core.** Register
  blocking is 2.16× on an M4 and 2.79× on an M1, but on Neoverse N2 the
  blocked `sdot` kernel runs at 73.9 against v1's 159.7 — also 2.16×, while
  on that core the *later* `smmla` kernel is the only thing that moves it
  further. Meanwhile Zen 3, with no VNNI and no `smmla`, gets nothing from
  either and keeps only the tile form's 1.39×.
* **Nothing here changed a score.** Every number in the table is the same
  checksum, which is what makes measuring at runtime a legitimate way to
  pick between these paths.

## On a real index

`examples/real_index.rs` scores a real next-plaid (≥1.7) index directory
against a `queries.npy`, with no dependencies (a small NPY reader is inside):

```
cargo run --release --example real_index -- <index_dir> <queries.npy> [n_queries] [ref_docs]
```

It reports exhaustive stage-2 time per query and ns/token on the real
residual codes, parity against float decompression (centroid + bucket weight
per dim, exact f32 MaxSim), and how loose the exact per-token skip bound
`max|w| · ‖q‖₁` is on real data. On SciFact / ColBERTv2 (5,183 docs, 1.25M
tokens, nbits 4) on an M4: 31.9 ns/token on real codes (same as synthetic),
max |Δscore| 0.0086 with mean relative error 1.5e-4 from the int8 query
quantisation, and the exact skip bound never fires (mean bound 0.70 against a
mean residual term of 0.039), so exact token skipping is not a lever here.

## Inside a real engine

The crate was wired into next-plaid 1.7.0 through its `Packing` trait, so both
routes read the same mmap index and the only difference is the kernel. Next to
them, a real v1.6.5 build — the release before the LUT kernel existed, which
decompressed every candidate to f32 and then scored it. SciFact (5,183 docs,
1.25M tokens, ColBERTv2, dim 128, nbits 4), 50 real queries, default search
parameters, Apple M4.

Forcing 1.7.0's float rescore path is *not* a stand-in for 1.6.5: it keeps
1.7.0's faster stage 1 and its improved gather, and understates the version gap
(it gives 2.5× where the real build gives 3.5×). Every 1.6.5 figure here comes
from a v1.6.5 worktree with the same stage instrumentation.

The harness is not in this repository: it is a patched next-plaid checkout with
stage timers, an `NpPacking` implementing this crate's `Packing` over
next-plaid's `ResidualCodec`, and an extra `ScoreQuery` arm selected by an
environment variable so both routes run interleaved in one process. Roughly 200
lines against next-plaid's own API, which is the point of the trait.

| build | stage 1 | stage 2 | end-to-end |
|---|---|---|---|
| next-plaid 1.6.5 | 5.01 ms | 14.22 ms | 19.26 ms |
| next-plaid 1.7.0 | 2.05 ms | 3.26 ms | 5.54 ms |
| 1.7.0 + this crate | 2.02 ms | 1.63 ms | 3.87 ms |

Ranking is unchanged by the swap: the two asymmetric arms return bit-identical
scores (0 of 50 top-10 lists differ, max relative delta 0.00e0), and both differ
from full decompression on 3 of 50 lists — the int8 query quantisation, which
this crate does not touch.

### Candidate depth

Stage 1 does not shrink with the shortlist, so the end-to-end gain is a function
of how many candidates you rescore (`n_full_scores/4` of them):

| candidates | 1.6.5 | 1.7.0 | + this crate | 1.6.5→1.7.0 | 1.6.5→crate |
|---|---|---|---|---|---|
| 64 | 8.79 ms | 2.33 | 2.22 | 3.78× | 3.96× |
| 128 | 13.05 | 2.63 | 2.35 | 4.95× | 5.54× |
| 256 | 13.28 | 3.06 | 2.55 | 4.33× | 5.20× |
| 512 | 14.12 | 3.96 | 3.03 | 3.56× | 4.65× |
| 1024 (default) | 19.73 | 5.61 | 3.78 | 3.52× | 5.23× |

So 1.6.5 → 1.7.0 is 3.5–5.0× depending on the working point, and 1.6.5 → this
crate is about 5× across the whole range. Quoting either as a single number
means picking a depth; the blog's 4.3–5.4× for 1.6.5 → 1.7.0 sits inside this
range, measured on a different checkpoint and with each version tuned to matched
quality rather than held to equal parameters.

Against 1.7.0 alone — the comparison that isolates this crate — the gain grows
from 1.08× at 64 candidates to 1.45× at the default 1024, because at shallow
depths stage 1 dominates and no stage-2 kernel can help. Measure your own split
before assuming this is your bottleneck.

### Why 1.6.5's curve is not a cost curve

Part of that 3.5–5.0× spread is scheduling, not arithmetic. 1.6.5 dispatched
rescoring in fixed 128-document chunks (`const DECOMPRESS_CHUNK_SIZE: usize =
128`, used as `par_chunks(...)`), so its per-document cost tracks how many
chunks the thread pool received:

| candidates | chunks | 1.6.5 µs/doc | 1.7.0 | + this crate |
|---|---|---|---|---|
| 64 | 1 | 63.0 | 5.4 | 2.8 |
| 128 | 1 | 64.0 | 4.4 | 2.1 |
| 256 | 2 | 32.5 | 3.6 | 1.8 |
| 512 | 4 | 17.4 | 3.4 | 1.8 |
| 1024 | 8 | 14.1 | 3.2 | 1.5 |

64 and 128 candidates cost the same per document because both are one chunk on
one thread; the figure halves as the chunk count doubles, then stops improving
at 8 chunks on a 10-core machine. The other two arms are near-flat, which is
what a cost curve should look like. 1.7.0 replaced the chunking with per-document
parallelism, so its advantage at shallow depths is partly that, not the kernel.
