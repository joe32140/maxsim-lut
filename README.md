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

So dispatch **calibrates**: on the first score of a process with more than
one executable kernel, each candidate scores a small synthetic document,
interleaved, and the winner is cached. It costs a few hundred microseconds
once. This is safe to decide at runtime precisely because every kernel is
bit-identical, so calibration can change how long a search takes and never
what it returns.

Override it when you need to:

```rust
let lut = lut.pin_kernel(Some(maxsim_lut::supported_kernels()[0])); // programmatic
```
```
MAXSIM_LUT_KERNEL=neon-i8mm     # pin by name
MAXSIM_LUT_NO_CALIBRATE=1       # take the first listed kernel, skip measuring
MAXSIM_LUT_FORCE_SCALAR=1       # the reference path
```

## Register blocking and tiles

Measured effect of the two structural changes over the straightforward
one-token, one-row-accumulator kernels this crate started from (32 query rows
× 240 tokens, dim 128, nbits 4, ns per scored token, bit-identical results):

| machine | before | register blocking | + GEMM tiles |
|---|---|---|---|
| Apple M4 (NEON sdot) | 70.6 | 32.7 | n/a |
| GitHub x86 runner (AVX-512 VNNI) | 143.8 | 86.3 | 37.2 |
| GitHub `ubuntu-24.04-arm` (NEON sdot) | 159.0 | 73.8 | n/a |

## Testing

```
cargo test
MAXSIM_LUT_FORCE_SCALAR=1 cargo test
```

Two tests pin the bit-exactness contract across every kernel the CPU can
execute, not just the one dispatch picks: the unit test
`every_supported_kernel_matches_scalar_bitwise` over a shape sweep, and the
integration test `pinned_kernels_agree_with_the_calibrated_default` through
the public pinning API. So an AVX-512 machine still verifies AVX2, and a
Neoverse core verifies both NEON kernels. CI runs x86_64 and aarch64 runners
plus a `cargo check` for a target with no SIMD path.

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
