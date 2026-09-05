//! aarch64 NEON kernel: `tbl` nibble expansion into plane order, `sdot`
//! accumulation, register-blocked over `NT` document tokens × 4 query rows.
//!
//! Blocking is why this beats a per-token loop: with one token at a time every
//! `sdot` needs two loads (a query chunk and a weight chunk), which caps the
//! core at ~1.5 `sdot`/cycle on a 3-load/cycle machine. Holding `NT` tokens'
//! expanded weights and 4 query rows in registers shares each query load
//! across `NT` products and each weight load across 4 rows: `(4 + NT)` loads
//! per `4·NT` `sdot`. It also gives the out-of-order core `4·NT` independent
//! accumulator chains instead of 4. The integer sums are identical to the
//! unblocked form (same products, exact integer addition), so the result is
//! bit-identical.
//!
//! `expand`, `fold4` and `fold_tail` are shared with the `smmla` kernel in
//! `neon_i8mm.rs`, which differs only in the dot instruction and the query
//! layout that instruction needs.

use std::arch::aarch64::*;

use super::Args;
use crate::lut::{Lut, NibbleTables};
use crate::MAX_DIM;

/// Document tokens per register block. 4 × 4 rows = 16 accumulators, well
/// inside the 32 NEON registers with 4 query and 1 weight chunk live.
const NT: usize = 4;

/// `sdot` via inline asm: pins the exact instruction independent of intrinsic
/// availability on the toolchain.
///
/// # Safety
/// Requires the `dotprod` CPU feature at runtime.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let out: int32x4_t;
    // SAFETY: pure register op; caller guarantees `dotprod`.
    unsafe {
        std::arch::asm!(
            "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) acc => out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    out
}

/// Fold four query rows' final integer accumulators (lane `r` = row
/// `base + r`) through the shared float tail. Bit-identical to the scalar
/// kernel on purpose: `vcvtq_f32_s32` is round-to-nearest like `as f32`;
/// multiply and add stay separate (never fused); `inv` multiplies last; and
/// for finite scores `vmaxq_f32(best, s)` equals the scalar `if s > best`.
///
/// # Safety
/// `base + 4 <= nq`; `crow` readable for `base + 4` f32s.
#[inline(always)]
pub(super) unsafe fn fold4(
    accv: int32x4_t,
    base: usize,
    sqw: &[f32],
    crow: *const f32,
    inv: f32,
    best: &mut [f32],
) {
    // SAFETY: see function contract.
    unsafe {
        let a = vcvtq_f32_s32(accv);
        let s = vmulq_f32(
            vaddq_f32(
                vmulq_f32(vld1q_f32(sqw.as_ptr().add(base)), a),
                vld1q_f32(crow.add(base)),
            ),
            vdupq_n_f32(inv),
        );
        let b = vld1q_f32(best.as_ptr().add(base));
        vst1q_f32(best.as_mut_ptr().add(base), vmaxq_f32(b, s));
    }
}

/// Scalar tail of the fold for `nq % 4` leftover rows; same expression, same
/// order as the vector form and the scalar kernel.
///
/// # Safety
/// `crow` readable for `accs.len()` f32s from its base.
#[inline(always)]
pub(super) unsafe fn fold_tail(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
    for i in 0..accs.len() {
        // SAFETY: see function contract.
        let c = unsafe { *crow.add(i) };
        let s = (sqw[i] * accs[i] as f32 + c) * inv;
        if s > best[i] {
            best[i] = s;
        }
    }
}

/// Expand one token's `pdim` packed bytes into `kpb` planes of int8 weights.
///
/// # Safety
/// `row.len() == pdim`, `kpb · pdim <= MAX_DIM`, tables loaded for `kpb` keys.
#[inline(always)]
pub(super) unsafe fn expand(
    row: &[u8],
    tabs: &[int8x16_t; 8],
    nib: &NibbleTables,
    kpb: usize,
    pdim: usize,
    w: &mut [i8; MAX_DIM],
) {
    // SAFETY: stores land in `[k·pdim, k·pdim + 16)` for `i + 16 <= pdim`,
    // inside the buffer; the tail copies only `rem` valid lanes.
    unsafe {
        let low_mask = vdupq_n_u8(0x0F);
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= pdim {
            let v = vld1q_u8(row.as_ptr().add(i));
            let hi = vshrq_n_u8(v, 4);
            let lo = vandq_u8(v, low_mask);
            for (k, tab) in tabs.iter().enumerate().take(kpb) {
                let idx = if nib.from_hi[k] { hi } else { lo };
                vst1q_s8(wp.add(k * pdim + i), vqtbl1q_s8(*tab, idx));
            }
            i += 16;
        }
        // Sub-16 tail through a zero-padded scratch so the store cannot clobber
        // the next plane's already-written low bytes. Keeps narrow dims (e.g.
        // dim 48 at nbits 2 = 12 bytes) on the SIMD path.
        if i < pdim {
            let rem = pdim - i;
            let mut src = [0u8; 16];
            src[..rem].copy_from_slice(&row[i..pdim]);
            let v = vld1q_u8(src.as_ptr());
            let hi = vshrq_n_u8(v, 4);
            let lo = vandq_u8(v, low_mask);
            let mut dst = [0i8; 16];
            for k in 0..kpb {
                let idx = if nib.from_hi[k] { hi } else { lo };
                vst1q_s8(dst.as_mut_ptr(), vqtbl1q_s8(tabs[k], idx));
                w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
            }
        }
    }
}

/// Score `N` expanded tokens against every query row, updating `best`.
///
/// # Safety
/// `dotprod`; `dim % 8 == 0`; query planes zero-padded to a multiple of 16
/// past `dim` (stride is a multiple of 64); `crows[j]` readable for `nq`
/// f32s; `best.len() == nq`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block<const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
    dim: usize,
    nq: usize,
    qp_base: *const i8,
    ps: usize,
    sqw: &[f32],
    crows: &[*const f32; N],
    invs: &[f32; N],
    best: &mut [f32],
) {
    // SAFETY: see function contract; every pointer stays inside the query
    // planes, the expanded buffers (reads of 16 lanes below `dim <= MAX_DIM`)
    // and the validated centroid rows.
    unsafe {
        let mut qi = 0usize;
        while qi + 4 <= nq {
            let mut acc = [[vdupq_n_s32(0); 4]; N];
            let qp = [
                qp_base.add(qi * ps),
                qp_base.add((qi + 1) * ps),
                qp_base.add((qi + 2) * ps),
                qp_base.add((qi + 3) * ps),
            ];
            let mut k = 0usize;
            while k < dim {
                let q = [
                    vld1q_s8(qp[0].add(k)),
                    vld1q_s8(qp[1].add(k)),
                    vld1q_s8(qp[2].add(k)),
                    vld1q_s8(qp[3].add(k)),
                ];
                for j in 0..N {
                    let wv = vld1q_s8(ws[j].as_ptr().add(k));
                    for r in 0..4 {
                        acc[j][r] = sdot(acc[j][r], q[r], wv);
                    }
                }
                k += 16;
            }
            for j in 0..N {
                // Pairwise tree -> [Σrow0, Σrow1, Σrow2, Σrow3] in one register.
                let accv = vpaddq_s32(vpaddq_s32(acc[j][0], acc[j][1]), vpaddq_s32(acc[j][2], acc[j][3]));
                fold4(accv, qi, sqw, crows[j], invs[j], best);
            }
            qi += 4;
        }
        if qi < nq {
            let rem = nq - qi;
            let mut tail = [0i32; 4];
            for j in 0..N {
                for (r, slot) in tail.iter_mut().enumerate().take(rem) {
                    let qp = qp_base.add((qi + r) * ps);
                    let mut acc0 = vdupq_n_s32(0);
                    let mut acc1 = vdupq_n_s32(0);
                    let mut k = 0usize;
                    while k < dim {
                        acc0 = sdot(acc0, vld1q_s8(qp.add(k)), vld1q_s8(ws[j].as_ptr().add(k)));
                        if k + 16 < dim {
                            acc1 = sdot(
                                acc1,
                                vld1q_s8(qp.add(k + 16)),
                                vld1q_s8(ws[j].as_ptr().add(k + 16)),
                            );
                        }
                        k += 32;
                    }
                    *slot = vaddvq_s32(vaddq_s32(acc0, acc1));
                }
                fold_tail(
                    &tail[..rem],
                    &sqw[qi..],
                    crows[j].add(qi),
                    invs[j],
                    &mut best[qi..],
                );
            }
        }
    }
}

/// # Safety
/// Requires `dotprod`; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables
/// present; query plane stride `>= padded_stride(dim)` (the dot loop reads
/// 16-byte chunks past `dim` into zero padding); every slice in `args`
/// validated by the scorer.
#[target_feature(enable = "dotprod")]
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
    let ps = q.stride();
    let qp_base = q.planes().as_ptr();
    let sqw = q.sqw();
    best.clear();
    best.resize(nq, f32::NEG_INFINITY);

    // SAFETY: preconditions above; `expand`/`block` contracts hold for every
    // token because the scorer validated `packed`, `codes`, `inv_norms`.
    unsafe {
        let mut tabs = [vdupq_n_s8(0); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = vld1q_s8(src.as_ptr());
        }
        let mut ws = [[0i8; MAX_DIM]; NT];
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
            block::<NT>(&ws, dim, nq, qp_base, ps, sqw, &crows, &invs, best);
            t += NT;
        }
        while t < a.n_tokens {
            expand(
                &a.packed[t * a.row_stride..t * a.row_stride + pdim],
                &tabs,
                nib,
                kpb,
                pdim,
                &mut ws[0],
            );
            let one_w = [ws[0]];
            let one_c = [a.crow(t)];
            let one_i = [a.inv(t)];
            block::<1>(&one_w, dim, nq, qp_base, ps, sqw, &one_c, &one_i, best);
            t += 1;
        }
    }
    best.iter().sum()
}
