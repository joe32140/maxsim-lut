//! Kernel dispatch and the scalar reference.
//!
//! Every SIMD kernel computes the identical integer accumulator per (query
//! row, doc token) and applies the identical float epilogue expression
//! `(sqw[q] · acc as f32 + crow[q]) · inv`, so all paths return the same
//! bits. The integration tests pin this.
//!
//! # Choosing among the kernels of one architecture
//!
//! Both supported architectures now carry more than one int8 dot
//! instruction, and which is fastest is a property of the core, not of the
//! instruction set:
//!
//! | | wide instruction | narrow instruction | who wins |
//! |---|---|---|---|
//! | aarch64 | `smmla`, 32 MACs | `sdot`, 16 MACs | Neoverse N2 issues both per cycle, so `smmla` doubles throughput; Apple M-series issues `smmla` at half rate and loses |
//! | x86_64 | `vpdpbusd` zmm, 64 MACs | `vpdpbusd` ymm / the AVX2 triple, 32 MACs | depends on how the part implements 512-bit ops and on how far it downclocks |
//!
//! Feature detection alone therefore cannot pick the fastest path: two cores
//! with identical feature bits disagree. [`select`] instead **measures**
//! them. On the first dispatch of a process that has more than one
//! executable kernel, [`calibrate`] scores a small synthetic document with
//! each candidate, interleaved, and caches the winner for the process.
//!
//! This is safe to do at runtime precisely because the kernels are
//! bit-identical: calibration can change how long a search takes, never what
//! it returns. It costs a few hundred microseconds, once. Set
//! `MAXSIM_LUT_NO_CALIBRATE=1` to take the first listed kernel instead, or
//! `MAXSIM_LUT_KERNEL=<name>` to pin one.

use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;

use crate::lut::Lut;
use crate::query::PreparedQuery;
use crate::scorer::Codes;
use crate::MAX_DIM;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx2_vnni;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
mod neon_i8mm;

/// Which code path scores a given shape on this CPU. Returned by
/// [`Lut::kernel`]; the `Display` form is meant for benchmark output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// The scalar reference, with the reason the SIMD paths were not taken.
    Scalar(ScalarReason),
    /// aarch64 NEON: `tbl` expansion, `sdot` accumulation.
    NeonSdot,
    /// aarch64 NEON + I8MM: `tbl` expansion, `smmla` 2×2 matrix accumulation
    /// (32 MACs per instruction). Faster than `sdot` on cores that issue
    /// both at the same rate (Arm Neoverse N2/V1/V2), not on Apple cores.
    NeonI8mm,
    /// x86_64 AVX2: `pshufb` expansion, `maddubs`/`madd` accumulation.
    Avx2,
    /// x86_64 AVX-VNNI: the AVX2 tile shape with 256-bit `vpdpbusd`, for
    /// cores that have VNNI without AVX-512 (Alder Lake and later hybrids).
    Avx2Vnni,
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
            Kernel::NeonI8mm => write!(f, "neon-i8mm"),
            Kernel::Avx2 => write!(f, "avx2"),
            Kernel::Avx2Vnni => write!(f, "avx2-vnni"),
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
        "neon-i8mm" => Some(Kernel::NeonI8mm),
        "avx2" => Some(Kernel::Avx2),
        "avx2-vnni" => Some(Kernel::Avx2Vnni),
        "avx512-vnni" => Some(Kernel::Avx512Vnni),
        _ => None,
    })
}

/// Every kernel this CPU can execute, in the order to try when calibration
/// is switched off (widest instruction first, which is the right guess on
/// most cores). Empty if the CPU has no SIMD path.
fn cpu_kernels() -> &'static [Kernel] {
    #[cfg(target_arch = "aarch64")]
    {
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        let i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        match (dotprod, i8mm) {
            (true, true) => &[Kernel::NeonI8mm, Kernel::NeonSdot],
            (true, false) => &[Kernel::NeonSdot],
            (false, true) => &[Kernel::NeonI8mm],
            (false, false) => &[],
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        let vnni256 = is_x86_feature_detected!("avxvnni");
        let avx2 = is_x86_feature_detected!("avx2");
        match (avx512, vnni256, avx2) {
            (true, true, _) => &[Kernel::Avx512Vnni, Kernel::Avx2Vnni, Kernel::Avx2],
            (true, false, _) => &[Kernel::Avx512Vnni, Kernel::Avx2],
            (false, true, _) => &[Kernel::Avx2Vnni, Kernel::Avx2],
            (false, false, true) => &[Kernel::Avx2],
            (false, false, false) => &[],
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    &[]
}

/// `MAXSIM_LUT_NO_CALIBRATE=1` skips the measurement and takes the first
/// entry of [`cpu_kernels`]. For reproducing a specific dispatch, and for
/// hosts that cannot spare the one-off cost at startup.
fn env_no_calibrate() -> bool {
    static SKIP: OnceLock<bool> = OnceLock::new();
    *SKIP.get_or_init(|| {
        std::env::var("MAXSIM_LUT_NO_CALIBRATE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// The fastest of this CPU's kernels on a representative shape, measured
/// once per process.
///
/// The shape (32 query rows × 64 document tokens, dim 128, nbits 4) is the
/// ColBERT working point; the arms are interleaved and reduced by minimum,
/// so a scheduling hiccup has to hit every repetition of one arm to change
/// the verdict. Any outcome is correct, only slower or faster, so there is
/// no failure mode to guard against: on any error building the probe the
/// first candidate is kept.
fn calibrated() -> Kernel {
    static CHOICE: OnceLock<Kernel> = OnceLock::new();
    *CHOICE.get_or_init(|| {
        let cands = cpu_kernels();
        match cands.first() {
            None => Kernel::Scalar(ScalarReason::CpuUnsupported),
            Some(&first) if cands.len() == 1 || env_no_calibrate() => first,
            Some(&first) => measure_fastest(cands).unwrap_or(first),
        }
    })
}

/// Time each candidate on a synthetic document; `None` if the probe could
/// not be built (which cannot happen for the constants below, but the
/// fallback keeps dispatch total).
fn measure_fastest(cands: &[Kernel]) -> Option<Kernel> {
    const DIM: usize = 128;
    const NQ: usize = 32;
    const NTOK: usize = 64;
    const REPS: usize = 5;
    const ITERS: usize = 8;

    let weights: Vec<f32> = (0..16).map(|i| -0.35 + 0.05 * i as f32).collect();
    let lut = Lut::colbert(4, &weights).ok()?;
    // Deterministic pseudo-random inputs: the kernels are data-independent
    // (no branches on values), so any well-mixed content times the same.
    let query: Vec<f32> = (0..NQ * DIM)
        .map(|i| ((i * 37 % 251) as f32 / 251.0) - 0.5)
        .collect();
    let q = PreparedQuery::new(&lut, &query, NQ, DIM).ok()?;
    let pdim = DIM / lut.keys_per_byte();
    let packed: Vec<u8> = (0..NTOK * pdim).map(|i| (i * 97 % 256) as u8).collect();
    let inv: Vec<f32> = (0..NTOK).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
    let args = Args {
        query: &q,
        packed: &packed,
        row_stride: pdim,
        n_tokens: NTOK,
        codes: Codes::None,
        cdot: q.zeros(),
        cdot_stride: 0,
        inv_norms: Some(&inv),
    };

    let mut best = vec![f64::INFINITY; cands.len()];
    for _ in 0..REPS {
        for (i, &k) in cands.iter().enumerate() {
            let t = Instant::now();
            for _ in 0..ITERS {
                black_box(run(k, &lut, &args));
            }
            let ns = t.elapsed().as_secs_f64();
            best[i] = best[i].min(ns);
        }
    }
    let mut winner = 0usize;
    for i in 1..cands.len() {
        if best[i] < best[winner] {
            winner = i;
        }
    }
    Some(cands[winner])
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
    // A pin from the host, then from the environment; both are honoured only
    // for a kernel this CPU can actually execute.
    for pin in [lut.pinned_kernel(), env_pin()].into_iter().flatten() {
        if cpu_kernels().contains(&pin) {
            return pin;
        }
    }
    calibrated()
}

/// Every SIMD kernel this CPU can execute, widest instruction first; empty
/// on a CPU or architecture with no SIMD path. The scalar reference always
/// runs everywhere and is not listed.
///
/// Pass one to [`crate::Lut::pin_kernel`] to bypass calibration.
pub fn supported_kernels() -> &'static [Kernel] {
    cpu_kernels()
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
/// AVX-512 machine still exercises the AVX2 path, and a core with both NEON
/// instructions exercises the one calibration did not pick; `select` alone
/// would run only the winner.
#[cfg(test)]
fn kernels_under_test() -> Vec<Kernel> {
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
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonI8mm => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: as above, with `i8mm` checked.
            unsafe { neon_i8mm::maxsim(lut, args, best, accs) }
        }),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: as above, with AVX2 checked.
            unsafe { avx2::maxsim(lut, args, best, accs) }
        }),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2Vnni => SCRATCH.with(|s| {
            let (best, accs) = &mut *s.borrow_mut();
            // SAFETY: as above, with avx2 + avxvnni checked.
            unsafe { avx2_vnni::maxsim(lut, args, best, accs) }
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
        let kernels = kernels_under_test();
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
