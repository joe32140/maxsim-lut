//! aarch64 NEON + I8MM kernel: `tbl` expansion, `smmla` matrix accumulation.
//!
//! `smmla Vd.4s, Vn.16b, Vm.16b` reads each 16-byte source as a 2×8 int8
//! matrix and accumulates the 2×2 product `Vn · Vmᵀ` into the four i32
//! lanes: `[n0·m0, n0·m1, n1·m0, n1·m1]`. With two query rows in `Vn`
//! ([`PreparedQuery::pairs`](crate::PreparedQuery)) and two document tokens'
//! weights in `Vm`, one instruction is 32 multiply-accumulates for a
//! 2-row × 2-token block, twice `sdot`'s 16.
//!
//! Whether that is faster depends on the core, not on the feature bit. Arm's
//! Neoverse N2/V1/V2 and Cortex-X2+ issue `smmla` at the same rate as `sdot`
//! and so double the dot throughput (measured 217 vs 109 G MAC/s on N2);
//! Apple M2–M4 issue it at half the rate and gain nothing (M4: 277 vs 243
//! in isolation, and below `sdot` once the interleaves are paid). Dispatch
//! therefore measures both once per process (`kernel::calibrate`) rather
//! than preferring one by feature.
//!
//! Register block: `RP` row pairs × `TP` token pairs, `RP·TP` accumulators;
//! per 8 dims `RP + TP` loads and `RP·TP` `smmla`. The default 4 × 2 covers
//! 8 rows × 4 tokens with 6 loads per 8 `smmla` and 14 live registers. The
//! two tokens of a pair are interleaved once per pair, right after
//! expansion, so the block loop itself is loads and `smmla` only.
//!
//! Exactness: the integer sums are the same products summed in a different
//! order, and the fold is `neon::fold4` / `fold_tail`, so results are
//! bit-identical to the scalar reference. An odd trailing token's pair
//! partner is all-zero weights and is never folded; an odd `nq`'s phantom
//! row is all-zero codes and is excluded by the fold's row count.

use std::arch::aarch64::*;

use super::neon::{expand, fold4, fold_tail};
use super::Args;
use crate::lut::Lut;
use crate::MAX_DIM;

/// Row pairs per block (8 query rows).
const RP: usize = 4;
/// Token pairs per block (4 document tokens).
const TP: usize = 2;
/// Bytes of one token pair's interleaved weights: `⌈dim/16⌉` 32-byte
/// double groups, at most `2 · MAX_DIM`.
const PAIR_BYTES: usize = 2 * MAX_DIM;

/// `smmla` via inline asm: the intrinsic is unstable on the MSRV toolchain.
///
/// # Safety
/// Requires the `i8mm` CPU feature at runtime.
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let out: int32x4_t;
    // SAFETY: pure register op; caller guarantees `i8mm`.
    unsafe {
        std::arch::asm!(
            "smmla {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) acc => out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    out
}

/// Interleave two tokens' expanded weights into 16-byte groups
/// `[w0[8g..8g+8], w1[8g..8g+8]]`, the `Vm` operand layout, for every
/// 8-dim group up to `dim` (rounded up to 16).
///
/// # Safety
/// `w0`, `w1` zero beyond `dim`; `dim <= MAX_DIM`.
#[inline(always)]
unsafe fn interleave(w0: &[i8; MAX_DIM], w1: &[i8; MAX_DIM], dim: usize, out: &mut [i8; PAIR_BYTES]) {
    // SAFETY: `k < dim <= 256` and `k % 16 == 0`, so reads of 16 lanes at `k`
    // and stores of 32 bytes at `2k` stay inside the buffers.
    unsafe {
        let mut k = 0usize;
        while k < dim {
            let a = vld1q_s8(w0.as_ptr().add(k));
            let b = vld1q_s8(w1.as_ptr().add(k));
            let o = out.as_mut_ptr().add(2 * k);
            vst1q_s8(o, vcombine_s8(vget_low_s8(a), vget_low_s8(b)));
            vst1q_s8(o.add(16), vcombine_s8(vget_high_s8(a), vget_high_s8(b)));
            k += 16;
        }
    }
}

/// One block: `R` row pairs (rows `row0 .. row0 + 2R`) × `N` token pairs
/// over all `d8n` dim groups, then the fold for those rows and the
/// `n_valid` real tokens (`2N − 1` or `2N`).
///
/// # Safety
/// `i8mm`; `qbase` points at row pair `row0/2` of the query's pair layout
/// with `pair_stride` bytes per pair and `R` pairs available (`row0 + 2(R−1)
/// < nq`); `wp[tp]` holds `16 · d8n` interleaved bytes; `crows[t]` readable
/// for `nq` f32s for `t < n_valid`; `sqw.len() == best.len() == nq`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
// `acc` is indexed `[row pair][token pair]` but the fold walks it token-pair
// outer, so the token index cannot be an iterator over `acc`.
#[allow(clippy::needless_range_loop)]
unsafe fn block<const R: usize, const N: usize>(
    wp: &[[i8; PAIR_BYTES]; N],
    n_valid: usize,
    d8n: usize,
    qbase: *const i8,
    pair_stride: usize,
    row0: usize,
    nq: usize,
    sqw: &[f32],
    crows: &[*const f32],
    invs: &[f32],
    best: &mut [f32],
) {
    // SAFETY: see function contract; the fold never touches a row `>= nq`
    // (the `take` bound) or a token `>= n_valid`.
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut acc = [[zero; N]; R];
        for g in 0..d8n {
            let mut q = [vdupq_n_s8(0); R];
            for (rp, qv) in q.iter_mut().enumerate() {
                *qv = vld1q_s8(qbase.add(rp * pair_stride + g * 16));
            }
            let mut w = [vdupq_n_s8(0); N];
            for (tp, wv) in w.iter_mut().enumerate() {
                *wv = vld1q_s8(wp[tp].as_ptr().add(g * 16));
            }
            for rp in 0..R {
                for tp in 0..N {
                    acc[rp][tp] = smmla(acc[rp][tp], q[rp], w[tp]);
                }
            }
        }
        // acc[rp][tp] lanes: [r0·t0, r0·t1, r1·t0, r1·t1] for the pair's rows
        // r0, r1 and the pair's tokens t0, t1. Two consecutive row pairs
        // unzip into one 4-row vector per token: even lanes for t0, odd for
        // t1. A lone final pair unzips against zero and folds 2 rows.
        for tp in 0..N {
            for half in 0..2 {
                let t = 2 * tp + half;
                if t >= n_valid {
                    break;
                }
                let (crow, inv) = (crows[t], invs[t]);
                let mut rp = 0usize;
                while rp < R {
                    let a = acc[rp][tp];
                    let paired = rp + 1 < R;
                    let b = if paired { acc[rp + 1][tp] } else { zero };
                    let v = if half == 0 {
                        vuzp1q_s32(a, b)
                    } else {
                        vuzp2q_s32(a, b)
                    };
                    let base = row0 + 2 * rp;
                    let take = (nq - base).min(if paired { 4 } else { 2 });
                    if take == 4 {
                        fold4(v, base, sqw, crow, inv, best);
                    } else {
                        let mut lanes = [0i32; 4];
                        vst1q_s32(lanes.as_mut_ptr(), v);
                        fold_tail(
                            &lanes[..take],
                            &sqw[base..],
                            crow.add(base),
                            inv,
                            &mut best[base..],
                        );
                    }
                    rp += 2;
                }
            }
        }
    }
}

/// All row pairs for `N` interleaved token pairs: blocks of `RP` pairs, then
/// the remainder.
///
/// # Safety
/// As [`block`], for the whole pair layout (`⌈nq/2⌉` pairs).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tokens<const N: usize>(
    wp: &[[i8; PAIR_BYTES]; N],
    n_valid: usize,
    d8n: usize,
    pairs: *const i8,
    pair_stride: usize,
    nq: usize,
    sqw: &[f32],
    crows: &[*const f32],
    invs: &[f32],
    best: &mut [f32],
) {
    // SAFETY: forwards the caller's contract block by block.
    unsafe {
        let npairs = nq.div_ceil(2);
        let mut p0 = 0usize;
        while npairs - p0 >= RP {
            let qb = pairs.add(p0 * pair_stride);
            block::<RP, N>(
                wp,
                n_valid,
                d8n,
                qb,
                pair_stride,
                2 * p0,
                nq,
                sqw,
                crows,
                invs,
                best,
            );
            p0 += RP;
        }
        let qb = pairs.add(p0 * pair_stride);
        match npairs - p0 {
            3 => block::<3, N>(
                wp,
                n_valid,
                d8n,
                qb,
                pair_stride,
                2 * p0,
                nq,
                sqw,
                crows,
                invs,
                best,
            ),
            2 => block::<2, N>(
                wp,
                n_valid,
                d8n,
                qb,
                pair_stride,
                2 * p0,
                nq,
                sqw,
                crows,
                invs,
                best,
            ),
            1 => block::<1, N>(
                wp,
                n_valid,
                d8n,
                qb,
                pair_stride,
                2 * p0,
                nq,
                sqw,
                crows,
                invs,
                best,
            ),
            _ => {}
        }
    }
}

/// # Safety
/// Requires `i8mm`; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables
/// present; every slice in `args` validated by the scorer.
#[target_feature(enable = "i8mm")]
pub(super) unsafe fn maxsim(lut: &Lut, a: &Args<'_>, best: &mut Vec<f32>, _accs: &mut Vec<i32>) -> f32 {
    let q = a.query;
    let nq = q.n_tokens();
    let dim = q.dim();
    if nq == 0 || a.n_tokens == 0 {
        return 0.0;
    }
    let nib = lut.nibble_tables().expect("select() guarantees nibble tables");
    let kpb = lut.keys_per_byte();
    let pdim = dim / kpb;
    let d8n = dim / 8;
    let pair_stride = 2 * q.stride();
    let pairs = q.pairs().as_ptr();
    debug_assert!(q.pairs().len() >= nq.div_ceil(2) * pair_stride);
    let sqw = q.sqw();
    best.clear();
    best.resize(nq, f32::NEG_INFINITY);

    // SAFETY: preconditions above; `expand`/`interleave`/`tokens` contracts
    // hold for every token because the scorer validated `packed`, `codes`,
    // `inv_norms`, and `PreparedQuery` built `⌈nq/2⌉ · pair_stride` bytes of
    // pairs. The expansion buffers are zero beyond `dim` (never written there).
    unsafe {
        let mut tabs = [vdupq_n_s8(0); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = vld1q_s8(src.as_ptr());
        }
        const NT: usize = 2 * TP;
        let zero = [0i8; MAX_DIM];
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut wp = [[0i8; PAIR_BYTES]; TP];
        let mut crows = [std::ptr::null::<f32>(); NT];
        let mut invs = [0f32; NT];
        let mut t = 0usize;
        while t + NT <= a.n_tokens {
            for j in 0..NT {
                let tt = t + j;
                expand(
                    &a.packed[tt * a.row_stride..tt * a.row_stride + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                crows[j] = a.crow(tt);
                invs[j] = a.inv(tt);
            }
            for tp in 0..TP {
                interleave(&ws[2 * tp], &ws[2 * tp + 1], dim, &mut wp[tp]);
            }
            tokens::<TP>(&wp, NT, d8n, pairs, pair_stride, nq, sqw, &crows, &invs, best);
            t += NT;
        }
        // Leftover tokens, one pair (or a final single) at a time.
        while t < a.n_tokens {
            let n = (a.n_tokens - t).min(2);
            for j in 0..n {
                let tt = t + j;
                expand(
                    &a.packed[tt * a.row_stride..tt * a.row_stride + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                crows[j] = a.crow(tt);
                invs[j] = a.inv(tt);
            }
            let partner = if n == 2 { &ws[1] } else { &zero };
            interleave(&ws[0], partner, dim, &mut wp[0]);
            let one = [wp[0]];
            tokens::<1>(
                &one,
                n,
                d8n,
                pairs,
                pair_stride,
                nq,
                sqw,
                &crows[..n],
                &invs[..n],
                best,
            );
            t += n;
        }
    }
    best.iter().sum()
}
