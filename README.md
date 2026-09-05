# maxsim-lut

Asymmetric int8-query × fused-LUT MaxSim over packed residual codes, the
stage-2 scoring kernel of a ColBERT / PLAID-style late-interaction engine,
packaged so any engine can call it. SIMD inside, slices outside, no
dependencies.

```text
q · token  =  q · centroid[cid]                   (optional; the host's stage-1 product)
           +  Σ_d q_d · bucket_weight[code_d]     (int8 query × int8 table, integer MACs)
score      =  Σ_q max_t (q · token_t) · inv_norm_t
```

The document is never decompressed to floats. Each packed residual byte is
turned directly into its `8/nbits` int8 bucket weights by an in-register
table lookup (NEON `tbl`, SSE `pshufb`), and the query-times-weights dot runs
on `sdot` / `smmla` / `maddubs` / `vpdpbusd`. Every SIMD path returns the
**same bits** as the scalar reference; the tests pin that on every shape.
Which of an architecture's kernels runs is decided by measuring them on the
host core, not by feature bits (see [per-architecture
specialisation](#per-architecture-specialisation)).

## Use

```rust
use maxsim_lut::{Codes, DocView, Lut, PreparedQuery, Scorer};

// Once per index: the codec's 2^nbits bucket weights and its packing layout.
let lut = Lut::colbert(nbits, &bucket_weights)?;

// Once per query: f32 tokens [n_tokens × dim], row-major.
let q = PreparedQuery::new(&lut, &query_f32, n_tokens, dim)?;

// Bind, optionally with the host's query×centroid scores, centroid-major
// [num_centroids × n_tokens].
let scorer = Scorer::new(&lut, &q).with_centroid_term(&cdot, num_centroids)?;

// Per candidate: borrowed slices, wherever the host keeps them (heap, mmap).
let doc = DocView::new(&packed_bytes, n_doc_tokens, row_stride)
    .codes(Codes::U32(&centroid_ids))
    .inv_norms(&inverse_norms);
let score: f32 = scorer.score(doc);

// Or many at once. Sequential: fan out with your own parallel iterator.
scorer.score_many(docs.iter().copied(), &mut scores);

println!("{}", lut.kernel(dim)); // e.g. "neon-sdot", "avx2-vnni", "scalar (DimNotSimdAligned)"
```

`Lut`, `PreparedQuery` and `Scorer` are `Send + Sync`. The crate owns no
thread pool: score across queries or across candidate chunks with whatever
your engine already uses.

## Preconditions

| requirement | why |
|---|---|
| `dim · nbits % 8 == 0`, `dim ≤ 256` | whole packed bytes; the expansion buffer is `[i8; 256]` |
| `dim % 8 == 0` for the SIMD paths | otherwise correct scores on the scalar path; `Lut::kernel(dim)` tells you |
| one packed row per token, contiguous, fixed `row_stride ≥ dim·nbits/8` | the kernel walks a document token-by-token |
| centroid term centroid-major `[num_centroids × n_query_tokens]` | one token's centroid row is then a single vector load |
| `nbits ∈ {1, 2, 4}` for SIMD | 8-bit codes have no 16-entry nibble table; they run scalar |

The win depends on the contiguity row. A host that scatters a document's
tokens across cells will see the kernel run and the speedup vanish.

## Packing

Hosts describe their bit layout with the `Packing` trait: for a byte and a
key position, which bucket index is stored. `ColbertPacking` ships the
ColBERTv2 / PLAID / fast-plaid / next-plaid / WARP layout. The 16-entry nibble
tables the SIMD paths use are derived from the packing and *verified* over
all 256 byte values at `Lut::new`; a layout they cannot express falls back
to the scalar path rather than diverging from it.

## Numbers

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
int8 dot instruction on the current core, which is what the kernel choice
below rests on.

## Per-architecture specialisation

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

Override it when you need to:

```rust
let lut = lut.pin_kernel(Some(maxsim_lut::supported_kernels()[0])); // programmatic
```
```
MAXSIM_LUT_KERNEL=neon-i8mm     # pin by name
MAXSIM_LUT_NO_CALIBRATE=1       # take the first listed kernel, skip measuring
MAXSIM_LUT_FORCE_SCALAR=1       # the reference path
```

## How much faster than the first version

Three structural changes separate the current kernels from the crate's first
commit (`aa2ecba`, the straight extraction from next-plaid): register
blocking, the GEMM-tile form on x86, and the per-core kernel choice above.
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

## Testing

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

`tests/concurrency.rs` is a separate binary, so its threads race for the
first dispatch in a cold process: it checks that concurrent scoring is
bit-identical to a single-threaded run, that every thread resolves
calibration to the same kernel, and that the public types are `Send + Sync`.
`the_self_check_rejects_a_kernel_that_disagrees` feeds calibration a runner
that lies about one kernel and asserts that exactly that kernel is dropped,
because an untested safety net is not a safety net.

CI runs x86_64 and aarch64 runners, a `cargo check` for a target with no SIMD
path, and the MSRV on both architectures.

### What has actually run on silicon

CI can only test the machines it is given, and GitHub's pool changes. As of
the last run:

| kernel | executed on |
|---|---|
| `neon-sdot` | Apple M1, Apple M4, Neoverse N2 |
| `neon-i8mm` | Neoverse N2, Apple M4 |
| `avx2` | AMD EPYC 7763, Intel Xeon |
| `avx512-vnni` | Intel Xeon (a runner class no longer in the pool) |
| `avx2-vnni` | **nothing yet** |

`avx2-vnni` is the one path no machine has run. Its two halves are covered
separately, which is the argument for shipping it: the 8-row tile addressing
and masked fold are the AVX2 kernel's, and `vpdpbusd` with the `128·Σw`
correction is the AVX-512 kernel's, both exercised on real hardware. What is
untested is the combination and the 256-bit VEX encoding, which is why the
runtime self-check exists. If you would rather not carry that risk, pin a
kernel or drop it from the candidate list with `Lut::pin_kernel`.

## Real data

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

## Provenance and license

The kernels are extracted from next-plaid's `residual_lut.rs`
([lightonai/next-plaid](https://github.com/lightonai/next-plaid), Apache-2.0),
with the codec coupling replaced by the `Packing` trait and ndarray by slices.
Apache-2.0; see `LICENSE` and `NOTICE`.
