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
on `sdot` / `maddubs` / `vpdpbusd`. Every SIMD path returns the **same bits**
as the scalar reference; the tests pin that on every shape.

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

println!("{}", lut.kernel(dim)); // e.g. "neon-sdot", "avx2", "scalar (DimNotSimdAligned)"
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

`cargo run --release --example bench` prints ns per scored token for the
dispatched kernel and the scalar reference and asserts they agree. Always read
the `kernel:` line before believing a number: a speedup attributed to a path
that never executed is the easiest measurement error to make. On Apple
Silicon with an x86 rustup default, pass `--target aarch64-apple-darwin`.

## Testing

```
cargo test
MAXSIM_LUT_FORCE_SCALAR=1 cargo test
```

The unit test `every_supported_kernel_matches_scalar_bitwise` runs every
kernel the CPU can execute, not just the one dispatch would pick, so an
AVX-512 machine also verifies the AVX2 path. `MAXSIM_LUT_KERNEL=avx2` (or
`neon-sdot`, `avx512-vnni`) pins dispatch to a supported kernel for
benchmarking it. CI runs on x86_64 (AVX-512 VNNI and AVX2) and aarch64 (NEON
dotprod), plus `cargo check` for a target with no SIMD path.

## Kernel shape

All three SIMD kernels expand a document token's packed bytes once (16-entry
in-register table per nibble) and then reuse those weights against every
query row, holding several tokens' weights and several rows in registers at
once. NEON keeps one accumulator per (row, token) and reduces with `sdot`;
the x86 kernels put 8 or 16 query rows into the lanes of one accumulator and
broadcast the weights 4 bytes at a time (`vpdpbusd` / `vpmaddubsw`), so no
horizontal reduction is ever needed. On AVX-512 the query is stored `+128` as
`u8` and the exact offset `128·Σw` is subtracted once per token.

## Provenance and license

The kernels are extracted from next-plaid's `residual_lut.rs`
([lightonai/next-plaid](https://github.com/lightonai/next-plaid), Apache-2.0),
with the codec coupling replaced by the `Packing` trait and ndarray by slices.
Apache-2.0; see `LICENSE` and `NOTICE`.
