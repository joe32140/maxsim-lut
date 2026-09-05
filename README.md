# maxsim-lut

The stage-2 scoring kernel of a ColBERT / PLAID-style late-interaction engine,
packaged so any engine can call it. SIMD inside, slices outside, no
dependencies.

It scores a candidate document's *stored* residual codes against a query
without decompressing the document to floats:

```text
q · token  =  q · centroid[cid]                   (optional; the host's stage-1 product)
           +  Σ_d q_d · bucket_weight[code_d]     (int8 query × int8 table, integer MACs)
score      =  Σ_q max_t (q · token_t) · inv_norm_t
```

Each packed byte becomes its `8/nbits` int8 bucket weights through an
in-register table lookup, and the dot runs on `sdot`, `smmla`, `maddubs` or
`vpdpbusd`. Every SIMD path returns the **same bits** as the scalar reference.

## Install

```
cargo add maxsim-lut
```

No dependencies, and none are added to your tree. MSRV 1.89.

## Use

```rust
use maxsim_lut::{Codes, DocView, Lut, PreparedQuery, Scorer};

// Once per index: the codec's 2^nbits bucket weights and its packing layout.
let lut = Lut::colbert(nbits, &bucket_weights)?;

// Once per query: f32 tokens [n_tokens × dim], row-major.
let q = PreparedQuery::new(&lut, &query_f32, n_tokens, dim)?;

// Bind, optionally with the host's query×centroid scores, centroid-major.
let scorer = Scorer::new(&lut, &q).with_centroid_term(&cdot, num_centroids)?;

// Per candidate: borrowed slices, wherever the host keeps them (heap, mmap).
let doc = DocView::new(&packed_bytes, n_doc_tokens, row_stride)
    .codes(Codes::U32(&centroid_ids))
    .inv_norms(&inverse_norms);
let score: f32 = scorer.score(doc);
```

`Lut`, `PreparedQuery` and `Scorer` are `Send + Sync`. The crate owns no thread
pool: parallelise across queries or candidate chunks however your engine
already does. The example on [docs.rs](https://docs.rs/maxsim-lut) compiles and
runs as written.

Hosts describe their bit layout through the `Packing` trait, and
`ColbertPacking` ships the ColBERTv2 / PLAID / fast-plaid / next-plaid / WARP
one.

## What it buys

Measured by wiring the crate into [next-plaid](https://github.com/lightonai/next-plaid)
1.7.0 and running SciFact (5,183 docs, 1.25M tokens, ColBERTv2, dim 128,
nbits 4), 50 real queries, default search parameters, Apple M4:

| build | stage 1 | stage 2 | end-to-end |
|---|---|---|---|
| next-plaid 1.6.5, which decompressed to f32 and then scored | 5.01 ms | 14.22 ms | 19.26 ms |
| next-plaid 1.7.0, the release that introduced this LUT kernel | 2.05 ms | 3.26 ms | 5.54 ms |
| **1.7.0 with its stage 2 replaced by this crate** | 2.02 ms | **1.63 ms** | **3.87 ms** |

Two separate steps, and only the second is this crate's. Getting the LUT kernel
at all (1.6.5 → 1.7.0) is 3.5–5.0× end-to-end depending on candidate depth.
Replacing that kernel with this one is a further **1.45× end-to-end, from 2.0×
on stage 2** — register blocking, a GEMM-tile form on x86, and picking the
kernel by measuring the host core.

Ranking is unchanged: the two asymmetric rows are bit-identical to each other.

Details, the candidate-depth sweep, and the v1-to-now history are in
[docs/benchmarks.md](docs/benchmarks.md).

## Preconditions

| requirement | why |
|---|---|
| `dim · nbits % 8 == 0`, `dim ≤ 256` | whole packed bytes; the expansion buffer is `[i8; 256]` |
| `dim % 8 == 0` for the SIMD paths | otherwise correct scores on the scalar path; `Lut::kernel(dim)` tells you |
| one packed row per token, contiguous, fixed `row_stride ≥ dim·nbits/8` | the kernel walks a document token-by-token |
| centroid term centroid-major `[num_centroids × n_query_tokens]` | one token's centroid row is then a single vector load |
| `nbits ∈ {1, 2, 4}` for SIMD | 8-bit codes have no 16-entry nibble table; they run scalar |

The win depends on the contiguity row. A host that scatters a document's tokens
across cells will see the kernel run and the speedup vanish.

## More

- [docs/kernels.md](docs/kernels.md) — the five kernels, why dispatch measures
  the host core instead of reading feature bits, the overrides, and which
  kernels have run on real silicon.
- [docs/benchmarks.md](docs/benchmarks.md) — how the numbers above were taken,
  the depth sweep, and what changed since the crate's first commit.
- [docs/testing.md](docs/testing.md) — how the bit-exactness contract is held.

## Background

The kernel and the problem it solves are described in [The slow half of
PLAID](https://chaochunhsu.github.io/blog/slow-half-of-plaid/), which measures
where a late-interaction query actually spends its time and why implicit
decompression is the lever. This crate is that idea extracted so engines other
than next-plaid can use it, and then taken further.

## Provenance and license

The kernels are extracted from next-plaid's `residual_lut.rs`
([lightonai/next-plaid](https://github.com/lightonai/next-plaid), Apache-2.0),
with the codec coupling replaced by the `Packing` trait and ndarray by slices.
Apache-2.0; see `LICENSE` and `NOTICE`.
