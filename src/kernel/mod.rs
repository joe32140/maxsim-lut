//! Kernel dispatch and the scalar reference.
//!
//! Every SIMD kernel computes the identical integer accumulator per (query
//! row, doc token) and applies the identical float epilogue expression
//! `(sqw[q] · acc as f32 + crow[q]) · inv`, so all paths return the same
//! bits. The integration tests pin this.

use std::sync::OnceLock;

use crate::lut::Lut;
use crate::query::PreparedQuery;
use crate::scorer::Codes;
use crate::MAX_DIM;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Which code path scores a given shape on this CPU. Returned by
/// [`Lut::kernel`]; the `Display` form is meant for benchmark output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// The scalar reference, with the reason the SIMD paths were not taken.
    Scalar(ScalarReason),
    /// aarch64 NEON: `tbl` expansion, `sdot` accumulation.
    NeonSdot,
    /// x86_64 AVX2: `pshufb` expansion, `maddubs`/`madd` accumulation.
    Avx2,
    /// x86_64 AVX-512 with VNNI: `pshufb` expansion, `vpdpbusd` accumulation.
    Avx512Vnni,
}

/// Why the scalar path runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarReason {
    /// [`Lut::force_scalar`] or `MAXSIM_LUT_FORCE_SCALAR=1`.
    Forced,
    /// `dim % 8 != 0` or `dim > MAX_DIM`.
    DimNotSimdAligned,
    /// The packing could not be factored into nibble tables (nbits 8).
    NoNibbleTables,
    /// This CPU lacks the required feature (NEON `dotprod`, AVX2), or the
    /// architecture has no SIMD path at all.
    CpuUnsupported,
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernel::Scalar(r) => write!(f, "scalar ({r:?})"),
            Kernel::NeonSdot => write!(f, "neon-sdot"),
            Kernel::Avx2 => write!(f, "avx2"),
            Kernel::Avx512Vnni => write!(f, "avx512-vnni"),
        }
    }
}

impl Kernel {
    /// `true` for any SIMD path.
    pub fn is_simd(&self) -> bool {
        !matches!(self, Kernel::Scalar(_))
    }
}

fn env_force_scalar() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| {
        std::env::var("MAXSIM_LUT_FORCE_SCALAR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// The dispatch decision, in one place, so [`Lut::kernel`] and
/// [`maxsim`] cannot disagree.
pub(crate) fn select(lut: &Lut, dim: usize) -> Kernel {
    if lut.force_scalar_set() || env_force_scalar() {
        return Kernel::Scalar(ScalarReason::Forced);
    }
    if !dim.is_multiple_of(8) || dim > MAX_DIM {
        return Kernel::Scalar(ScalarReason::DimNotSimdAligned);
    }
    if lut.nibble_tables().is_none() {
        return Kernel::Scalar(ScalarReason::NoNibbleTables);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Kernel::NeonSdot;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            return Kernel::Avx512Vnni;
        }
        if is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
    }
    Kernel::Scalar(ScalarReason::CpuUnsupported)
}

/// Everything a kernel needs, already validated by [`crate::Scorer`].
pub(crate) struct Args<'a> {
    pub query: &'a PreparedQuery,
    /// `[n_tokens · row_stride]` (at least) packed residual bytes.
    pub packed: &'a [u8],
    pub row_stride: usize,
    pub n_tokens: usize,
    /// Centroid ids; `Codes::None` iff `cdot_stride == 0`.
    pub codes: Codes<'a>,
    /// Centroid-major `[num_centroids · nq]` scores, or `[nq]` zeros with stride 0.
    pub cdot: &'a [f32],
    pub cdot_stride: usize,
    pub inv_norms: Option<&'a [f32]>,
}

impl Args<'_> {
    #[inline(always)]
    pub(crate) fn inv(&self, t: usize) -> f32 {
        match self.inv_norms {
            Some(v) => v[t],
            None => 1.0,
        }
    }
    #[inline(always)]
    pub(crate) fn crow(&self, t: usize) -> *const f32 {
        // Range was validated by the scorer; stride 0 selects the zero row.
        let off = self.codes.id(t) * self.cdot_stride;
        debug_assert!(off + self.query.n_tokens() <= self.cdot.len());
        // SAFETY of the later reads: off + nq <= cdot.len() (validated).
        unsafe { self.cdot.as_ptr().add(off) }
    }
}

// Per-thread kernel scratch (`best`, `accs`), reused across the thousands of
// per-candidate calls of a search. The kernels size-and-initialise it on
// entry, so no state leaks between calls.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
thread_local! {
    static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<i32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Runtime-dispatched MaxSim. `args` must come from `Scorer::validate`.
pub(crate) fn maxsim(lut: &Lut, args: &Args<'_>) -> f32 {
    let dim = args.query.dim();
    match select(lut, dim) {
        Kernel::Scalar(_) => scalar(lut, args),
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonSdot => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: `select` checked `dotprod`, dim % 8 == 0, dim <= MAX_DIM,
            // nibble tables present; the scorer validated every slice length.
            unsafe { neon::maxsim(lut, args, best, accs) }
        }),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: as above, with AVX2 checked.
            unsafe { avx2::maxsim(lut, args, best, accs) }
        }),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512Vnni => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: as above, with avx512f/bw/vnni checked.
            unsafe { avx512::maxsim(lut, args, best, accs) }
        }),
        #[allow(unreachable_patterns)]
        _ => scalar(lut, args),
    }
}

/// Scalar reference. Doc-token-outer: expand each stored token's bytes to
/// int8 weights once, amortised over all query rows.
pub(crate) fn scalar(lut: &Lut, a: &Args<'_>) -> f32 {
    let q = a.query;
    let nq = q.n_tokens();
    let dim = q.dim();
    if nq == 0 || a.n_tokens == 0 {
        return 0.0;
    }
    let kpb = lut.keys_per_byte();
    let pdim = dim / kpb;
    let qv = q.codes();
    let sqw = q.sqw();
    let mut best = vec![f32::NEG_INFINITY; nq];
    let mut w = [0i8; MAX_DIM];
    for t in 0..a.n_tokens {
        let row = &a.packed[t * a.row_stride..t * a.row_stride + pdim];
        for (i, &byte) in row.iter().enumerate() {
            w[i * kpb..(i + 1) * kpb].copy_from_slice(lut.expand(byte));
        }
        let inv = a.inv(t);
        let crow = a.crow(t);
        for (qi, best_q) in best.iter_mut().enumerate() {
            let qrow = &qv[qi * dim..(qi + 1) * dim];
            let mut acc = 0i32;
            for (qd, wd) in qrow.iter().zip(&w[..dim]) {
                acc += *qd as i32 * *wd as i32;
            }
            // SAFETY: crow points at >= nq readable f32s (validated).
            let c = unsafe { *crow.add(qi) };
            let score = (sqw[qi] * acc as f32 + c) * inv;
            if score > *best_q {
                *best_q = score;
            }
        }
    }
    best.iter().sum()
}
