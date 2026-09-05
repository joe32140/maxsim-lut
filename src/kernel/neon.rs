//! aarch64 NEON kernel: `tbl` nibble expansion into plane order, `sdot`
//! accumulation, four-row vectorised epilogue.

use std::arch::aarch64::*;

use super::Args;
use crate::lut::Lut;
use crate::MAX_DIM;

/// `sdot` via inline asm: the intrinsic needs the `dotprod` target feature
/// on the enclosing function, which we have, but the asm form documents the
/// exact instruction and pins it independent of intrinsic availability.
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
unsafe fn fold4(accv: int32x4_t, base: usize, sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
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
unsafe fn fold_tail(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
    for i in 0..accs.len() {
        // SAFETY: see function contract.
        let c = unsafe { *crow.add(i) };
        let s = (sqw[i] * accs[i] as f32 + c) * inv;
        if s > best[i] {
            best[i] = s;
        }
    }
}

/// # Safety
/// Requires `dotprod`; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables
/// present; query plane stride `>= padded_stride(dim)` (the dot loop reads
/// 16-byte chunks past `dim` into zero padding); every slice in `args`
/// validated by the scorer.
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn maxsim(lut: &Lut, a: &Args<'_>, best: &mut Vec<f32>, accs: &mut Vec<i32>) -> f32 {
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
    accs.clear();
    accs.resize(nq, 0);
    let mut w = [0i8; MAX_DIM];

    // SAFETY: all pointer arithmetic below stays within slices the scorer
    // validated, the `[i8; MAX_DIM]` buffer (dim <= 256, chunks <= 32 lanes
    // past a multiple-of-32 offset below dim), and the zero-padded query
    // planes (stride is a multiple of 64).
    unsafe {
        let mut tabs = [vdupq_n_s8(0); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = vld1q_s8(src.as_ptr());
        }
        let low_mask = vdupq_n_u8(0x0F);

        for t in 0..a.n_tokens {
            let row = &a.packed[t * a.row_stride..t * a.row_stride + pdim];
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
            // Sub-16 tail: expand through a zero-padded 16-byte scratch and
            // copy out only the valid lanes, so the store cannot clobber the
            // next plane's already-written low bytes. Keeps narrow dims
            // (e.g. dim 48 at nbits 2 = 12 bytes) on the SIMD path.
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
            let wp = w.as_ptr();
            let inv = a.inv(t);
            let crow = a.crow(t);

            // Per-row SDOT over the shared expanded weights, two independent
            // accumulators. Partial tail chunks are exact: both sides zero-pad.
            macro_rules! row_acc {
                ($qi:expr) => {{
                    let qp = qp_base.add($qi * ps);
                    let mut acc0 = vdupq_n_s32(0);
                    let mut acc1 = vdupq_n_s32(0);
                    let mut k = 0usize;
                    while k < dim {
                        acc0 = sdot(acc0, vld1q_s8(qp.add(k)), vld1q_s8(wp.add(k)));
                        if k + 16 < dim {
                            acc1 = sdot(acc1, vld1q_s8(qp.add(k + 16)), vld1q_s8(wp.add(k + 16)));
                        }
                        k += 32;
                    }
                    vaddq_s32(acc0, acc1)
                }};
            }
            let mut qi = 0usize;
            while qi + 4 <= nq {
                let v0 = row_acc!(qi);
                let v1 = row_acc!(qi + 1);
                let v2 = row_acc!(qi + 2);
                let v3 = row_acc!(qi + 3);
                // Pairwise tree -> [Σv0, Σv1, Σv2, Σv3] in one register.
                let accv = vpaddq_s32(vpaddq_s32(v0, v1), vpaddq_s32(v2, v3));
                fold4(accv, qi, sqw, crow, inv, best);
                qi += 4;
            }
            if qi < nq {
                let rem = nq - qi;
                for (r, acc) in accs.iter_mut().enumerate().take(rem) {
                    *acc = vaddvq_s32(row_acc!(qi + r));
                }
                fold_tail(&accs[..rem], &sqw[qi..], crow.add(qi), inv, &mut best[qi..]);
            }
        }
    }
    best.iter().sum()
}
