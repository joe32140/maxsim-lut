//! Asymmetric int8-query × fused-LUT MaxSim over packed residual codes.
//!
//! This is the stage-2 scoring kernel of a ColBERT / PLAID-style late
//! interaction engine, extracted so any engine can call it: score one
//! candidate document's *stored* residual codes against a query, without
//! decompressing the document to floats first.
//!
//! ```text
//! q · token  =  q · centroid[cid]                    (optional, supplied by the host)
//!            +  Σ_d q_d · bucket_weight[code_d]      (int8 query × int8 table, integer MACs)
//! score      =  Σ_q max_t (q · token_t) · inv_norm_t (optional per-token normalisation)
//! ```
//!
//! The document side is a table that turns each packed residual *byte*
//! straight into its `8/nbits` int8 bucket weights ([`Lut`]). The query side
//! is int8 codes with one f32 scale per row ([`PreparedQuery`]). A
//! [`Scorer`] binds the two, plus the host's optional centroid term, and
//! scores [`DocView`]s: borrowed slices pointing at wherever the host keeps
//! its bytes (heap, mmap, cell-contiguous).
//!
//! # What is inside, what is outside
//!
//! Inside: the loop order (doc-token-outer, so each token expands once and is
//! reused across every query row), the SIMD (NEON `sdot` and `smmla`, AVX2,
//! AVX-VNNI and AVX-512 VNNI), the runtime dispatch, and a scalar reference
//! every SIMD path must match **bit-for-bit**.
//!
//! Where an architecture offers more than one int8 dot instruction, the
//! faster one depends on the core rather than on the feature bits: Arm's
//! Neoverse N2 doubles its throughput with `smmla` while Apple's M4 loses
//! with it. Dispatch therefore measures the candidates once per process and
//! caches the winner ([`supported_kernels`], [`Lut::pin_kernel`]). Because
//! every kernel is bit-identical, that choice can only change speed.
//!
//! Outside: candidate generation, IVF, storage, threads. The crate has no
//! dependencies and owns no thread pool; [`Lut`], [`PreparedQuery`] and
//! [`Scorer`] are `Send + Sync`, so the host parallelises across queries or
//! across candidate chunks however it already does.
//!
//! # Example
//!
//! ```
//! use maxsim_lut::{ColbertPacking, Codes, DocView, Lut, PreparedQuery, Scorer};
//!
//! let dim = 128;
//! let nbits = 4;
//! // Bucket weights come from the host's residual quantiser (2^nbits of them).
//! let weights: Vec<f32> = (0..16).map(|i| -0.3 + 0.04 * i as f32).collect();
//! let lut = Lut::new(&ColbertPacking::new(nbits).unwrap(), &weights).unwrap();
//!
//! // One query: 32 tokens of dim 128, row-major f32.
//! let query = vec![0.01f32; 32 * dim];
//! let q = PreparedQuery::new(&lut, &query, 32, dim).unwrap();
//!
//! // Stage-1 product the host already has: [num_centroids, n_query_tokens].
//! let num_centroids = 1024;
//! let cdot = vec![0.0f32; num_centroids * 32];
//! let scorer = Scorer::new(&lut, &q).with_centroid_term(&cdot, num_centroids).unwrap();
//!
//! // A candidate document: 200 tokens, 64 packed bytes each, one centroid id per token.
//! let packed = vec![0u8; 200 * 64];
//! let codes = vec![7u32; 200];
//! let doc = DocView::new(&packed, 200, 64).codes(Codes::U32(&codes));
//! let score: f32 = scorer.score(doc);
//! assert!(score.is_finite());
//! println!("kernel in use: {}", lut.kernel(dim));
//! ```
//!
//! # Preconditions the host must know
//!
//! * `dim · nbits` must be a multiple of 8 (whole packed bytes) and `dim ≤ 256`.
//! * The SIMD paths need `dim % 8 == 0`; other dims score correctly on the
//!   scalar path. [`Lut::kernel`] tells you which path will run.
//! * A document's packed rows must be contiguous, one row per token, at a
//!   fixed `row_stride` of at least `dim / (8/nbits)` bytes.
//! * The win depends on that contiguity. A host that scatters a document's
//!   tokens across cells will see the kernel run and the speedup vanish.
//!
//! # Provenance
//!
//! The kernels are extracted from next-plaid's `residual_lut.rs`
//! (Apache-2.0, <https://github.com/lightonai/next-plaid>), with the codec
//! coupling replaced by the [`Packing`] trait and the ndarray types by slices.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod kernel;
mod lut;
mod packing;
mod query;
mod scorer;

pub use kernel::{supported_kernels, warm_up, Kernel, ScalarReason};
pub use lut::{Lut, NibbleTables, MAX_DIM};
pub use packing::{ColbertPacking, Packing};
pub use query::PreparedQuery;
pub use scorer::{Codes, DocView, Scorer};

/// Errors from building tables and queries or from shape validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `nbits` must be 1, 2, 4 or 8.
    NbitsUnsupported(usize),
    /// Expected `2^nbits` bucket weights.
    BucketCount {
        /// `2^nbits`.
        expected: usize,
        /// What was passed.
        got: usize,
    },
    /// `dim · nbits` is not a multiple of 8, so rows would not be whole bytes.
    DimNotByteAligned {
        /// The embedding dimension.
        dim: usize,
        /// The code width.
        nbits: usize,
    },
    /// `dim` exceeds [`MAX_DIM`].
    DimTooLarge(usize),
    /// A slice length or stride does not match the declared shape.
    Shape(String),
    /// The query was prepared against a different table.
    LutMismatch,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NbitsUnsupported(n) => write!(f, "nbits must be 1, 2, 4 or 8, got {n}"),
            Error::BucketCount { expected, got } => {
                write!(f, "expected {expected} bucket weights (2^nbits), got {got}")
            }
            Error::DimNotByteAligned { dim, nbits } => {
                write!(f, "dim {dim} × nbits {nbits} is not a multiple of 8")
            }
            Error::DimTooLarge(d) => write!(f, "dim {d} exceeds MAX_DIM {MAX_DIM}"),
            Error::Shape(s) => write!(f, "shape mismatch: {s}"),
            Error::LutMismatch => write!(f, "query was prepared against a different Lut"),
        }
    }
}

impl std::error::Error for Error {}

/// Row stride, in lanes, that the SIMD kernels read query rows at: `dim`
/// rounded up to 64 lanes, a multiple of the NEON (16), AVX2 (32) and
/// AVX-512 (64) chunk widths. Padding lanes are zero and contribute
/// `anything · 0 = 0`.
#[inline]
pub(crate) fn padded_stride(dim: usize) -> usize {
    dim.div_ceil(64) * 64
}
