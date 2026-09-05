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

/// `MAXSIM_LUT_KERNEL=<name>` pins a SIMD kernel by its `Display` name
/// (`neon-sdot`, `avx2`, `avx512-vnni`), for benchmarking a path the CPU
/// would not dispatch to by default (e.g. AVX2 on an AVX-512 machine). It is
/// honoured only when the CPU supports that kernel and the shape is SIMD
/// eligible; otherwise dispatch proceeds normally. Read once per process.
fn env_pin() -> Option<Kernel> {
    static PIN: OnceLock<Option<Kernel>> = OnceLock::new();
    *PIN.get_or_init(|| match std::env::var("MAXSIM_LUT_KERNEL").ok()?.as_str() {
        "neon-sdot" => Some(Kernel::NeonSdot),
        "avx2" => Some(Kernel::Avx2),
        "avx512-vnni" => Some(Kernel::Avx512Vnni),
        _ => None,
    })
}

/// What the CPU can execute, best first (`None` if nothing SIMD).
fn cpu_kernels() -> &'static [Kernel] {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return &[Kernel::NeonSdot];
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        if avx512 {
            return &[Kernel::Avx512Vnni, Kernel::Avx2];
        }
        if is_x86_feature_detected!("avx2") {
            return &[Kernel::Avx2];
        }
    }
    &[]
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
    let cpu = cpu_kernels();
    if let Some(pin) = env_pin() {
        if cpu.contains(&pin) {
            return pin;
        }
    }
    cpu.first()
        .copied()
        .unwrap_or(Kernel::Scalar(ScalarReason::CpuUnsupported))
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
    run(select(lut, args.query.dim()), lut, args)
}

/// Every kernel this CPU can execute, scalar first. Used by the tests so an
/// AVX-512 machine still exercises the AVX2 path; `select` alone would skip it.
#[cfg(test)]
pub(crate) fn supported_kernels() -> Vec<Kernel> {
    let mut v = vec![Kernel::Scalar(ScalarReason::Forced)];
    v.extend_from_slice(cpu_kernels());
    v
}

/// Run a specific kernel. `kernel` must be executable on this CPU and, for
/// the SIMD variants, `select` must not have returned a `Scalar` reason
/// other than `Forced` for this `lut`/`dim` (i.e. dim aligned, tables present).
pub(crate) fn run(kernel: Kernel, lut: &Lut, args: &Args<'_>) -> f32 {
    match kernel {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColbertPacking;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f32(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
        }
    }

    /// Every executable kernel agrees bitwise with the scalar reference. This
    /// is the test that reaches AVX2 on an AVX-512 host and AVX-512 on any
    /// host that has it; the integration tests only see whatever `select`
    /// picks.
    #[test]
    fn every_supported_kernel_matches_scalar_bitwise() {
        let kernels = supported_kernels();
        let mut rng = Rng(0x452821E638D01377);
        for &nq in &[1usize, 3, 7, 8, 9, 16, 17, 32] {
            for &nbits in &[1usize, 2, 4] {
                for &dim in &[8usize, 16, 40, 48, 96, 128, 200, 256] {
                    let p = ColbertPacking::new(nbits).unwrap();
                    let n = 1usize << nbits;
                    let mut w: Vec<f32> = (0..n).map(|_| rng.f32(-0.4, 0.4)).collect();
                    w.sort_by(|a, b| a.total_cmp(b));
                    let lut = Lut::new(&p, &w).unwrap();
                    let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
                    let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
                    let ntok = 11;
                    let pdim = dim / lut.keys_per_byte();
                    let row_stride = pdim + 3;
                    let packed: Vec<u8> = (0..ntok * row_stride).map(|_| (rng.next() >> 56) as u8).collect();
                    let ncent = 5;
                    let codes: Vec<u32> = (0..ntok).map(|_| (rng.next() % ncent as u64) as u32).collect();
                    let cdot: Vec<f32> = (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect();
                    let inv: Vec<f32> = (0..ntok).map(|_| rng.f32(0.5, 1.5)).collect();
                    for with_cdot in [false, true] {
                        for with_inv in [false, true] {
                            let args = Args {
                                query: &q,
                                packed: &packed,
                                row_stride,
                                n_tokens: ntok,
                                codes: if with_cdot {
                                    Codes::U32(&codes)
                                } else {
                                    Codes::None
                                },
                                cdot: if with_cdot { &cdot } else { q.zeros() },
                                cdot_stride: if with_cdot { nq } else { 0 },
                                inv_norms: if with_inv { Some(&inv) } else { None },
                            };
                            let want = scalar(&lut, &args);
                            for &k in &kernels {
                                let got = run(k, &lut, &args);
                                assert_eq!(
                                    got.to_bits(),
                                    want.to_bits(),
                                    "{k}: nq {nq} nbits {nbits} dim {dim} cdot {with_cdot} inv {with_inv}: {got} vs {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!("kernels exercised: {:?}", kernels);
    }
}
