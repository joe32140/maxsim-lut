//! x86_64 AVX-512 + VNNI kernel, register-blocked over `NT` document tokens
//! × 4 query rows (see `neon.rs` for why blocking pays).
//!
//! The expand stays 128-bit `pshufb` (charged once per doc token and
//! amortised over every query row); the per-pair work is 512-bit. The dot
//! uses `vpdpbusd` (unsigned × signed): we feed it `|w|` against `sign(w)·q`.
//! `|w|` and w's sign mask are computed once per weight chunk and shared by
//! the 4 rows; the per-pair cost is one masked negate plus one `vpdpbusd`.
//! `-128` never occurs on either side (both clamped to ±127), so the negation
//! is exact, and `w == 0` lanes zero the product regardless.

use std::arch::x86_64::*;

use super::Args;
use crate::lut::{Lut, NibbleTables};
use crate::MAX_DIM;

/// Tokens per register block: 16 zmm accumulators + 4 query + 3 weight-side
/// registers fits the 32 zmm file.
const NT: usize = 4;

/// Sixteen-row fold; same op order as the AVX2/NEON/scalar forms.
///
/// # Safety
/// `crow` readable for `accs.len()` f32s.
#[inline(always)]
unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
    let nq = accs.len();
    // SAFETY: see function contract; vector loads stay within `i + 16 <= nq`.
    unsafe {
        let invv = _mm512_set1_ps(inv);
        let mut i = 0usize;
        while i + 16 <= nq {
            let a = _mm512_cvtepi32_ps(_mm512_loadu_si512(accs.as_ptr().add(i) as *const _));
            let s = _mm512_mul_ps(
                _mm512_add_ps(
                    _mm512_mul_ps(_mm512_loadu_ps(sqw.as_ptr().add(i)), a),
                    _mm512_loadu_ps(crow.add(i)),
                ),
                invv,
            );
            let b = _mm512_loadu_ps(best.as_ptr().add(i));
            _mm512_storeu_ps(best.as_mut_ptr().add(i), _mm512_max_ps(b, s));
            i += 16;
        }
        while i < nq {
            let s = (sqw[i] * accs[i] as f32 + *crow.add(i)) * inv;
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
    }
}

/// Expand one token's `pdim` packed bytes into `kpb` planes of int8 weights.
///
/// # Safety
/// `row.len() == pdim`, `kpb · pdim <= MAX_DIM`, tables loaded for `kpb` keys.
#[inline(always)]
unsafe fn expand(
    row: &[u8],
    tabs: &[__m128i; 8],
    nib: &NibbleTables,
    kpb: usize,
    pdim: usize,
    w: &mut [i8; MAX_DIM],
) {
    // SAFETY: stores land in `[k·pdim, k·pdim + 16)` for `i + 16 <= pdim`;
    // the tail copies only `rem` valid lanes.
    unsafe {
        let low_mask = _mm_set1_epi8(0x0F);
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= pdim {
            let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
            let lo = _mm_and_si128(v, low_mask);
            for (k, tab) in tabs.iter().enumerate().take(kpb) {
                let idx = if nib.from_hi[k] { hi } else { lo };
                _mm_storeu_si128(wp.add(k * pdim + i) as *mut __m128i, _mm_shuffle_epi8(*tab, idx));
            }
            i += 16;
        }
        if i < pdim {
            let rem = pdim - i;
            let mut src = [0u8; 16];
            src[..rem].copy_from_slice(&row[i..pdim]);
            let v = _mm_loadu_si128(src.as_ptr() as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
            let lo = _mm_and_si128(v, low_mask);
            let mut dst = [0i8; 16];
            for k in 0..kpb {
                let idx = if nib.from_hi[k] { hi } else { lo };
                _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, _mm_shuffle_epi8(tabs[k], idx));
                w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
            }
        }
    }
}

/// Integer accumulators for `N` expanded tokens × all `nq` rows into
/// `accs[j·nq + qi]`, then fold each token.
///
/// # Safety
/// AVX-512F/BW/VNNI; `dim % 8 == 0`; query planes zero-padded to a multiple
/// of 64 past `dim`; `crows[j]` readable for `nq` f32s; `accs.len() >= N·nq`;
/// `best.len() == nq`.
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
    accs: &mut [i32],
    best: &mut [f32],
) {
    // SAFETY: see function contract; 64-lane reads below a multiple-of-64
    // bound <= 256 stay inside the expanded buffers and padded planes.
    unsafe {
        let zero = _mm512_setzero_si512();
        let mut qi = 0usize;
        while qi + 4 <= nq {
            let mut acc = [[zero; 4]; N];
            let qp = [
                qp_base.add(qi * ps),
                qp_base.add((qi + 1) * ps),
                qp_base.add((qi + 2) * ps),
                qp_base.add((qi + 3) * ps),
            ];
            let mut k = 0usize;
            while k < dim {
                let q = [
                    _mm512_loadu_si512(qp[0].add(k) as *const _),
                    _mm512_loadu_si512(qp[1].add(k) as *const _),
                    _mm512_loadu_si512(qp[2].add(k) as *const _),
                    _mm512_loadu_si512(qp[3].add(k) as *const _),
                ];
                for j in 0..N {
                    let wv = _mm512_loadu_si512(ws[j].as_ptr().add(k) as *const _);
                    let mag = _mm512_abs_epi8(wv);
                    let neg = _mm512_movepi8_mask(wv);
                    for r in 0..4 {
                        let sq = _mm512_mask_sub_epi8(q[r], neg, zero, q[r]);
                        acc[j][r] = _mm512_dpbusd_epi32(acc[j][r], mag, sq);
                    }
                }
                k += 64;
            }
            for j in 0..N {
                for r in 0..4 {
                    accs[j * nq + qi + r] = _mm512_reduce_add_epi32(acc[j][r]);
                }
            }
            qi += 4;
        }
        while qi < nq {
            let qp = qp_base.add(qi * ps);
            for j in 0..N {
                let mut acc = zero;
                let mut k = 0usize;
                while k < dim {
                    let qv = _mm512_loadu_si512(qp.add(k) as *const _);
                    let wv = _mm512_loadu_si512(ws[j].as_ptr().add(k) as *const _);
                    let sq = _mm512_mask_sub_epi8(qv, _mm512_movepi8_mask(wv), zero, qv);
                    acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), sq);
                    k += 64;
                }
                accs[j * nq + qi] = _mm512_reduce_add_epi32(acc);
            }
            qi += 1;
        }
        for j in 0..N {
            fold_block(&accs[j * nq..(j + 1) * nq], sqw, crows[j], invs[j], best);
        }
    }
}

/// # Safety
/// Requires `avx512f,avx512bw,avx512vnni`; `dim % 8 == 0 && dim <= MAX_DIM`;
/// nibble tables present; query plane stride a multiple of 64 (64-byte
/// chunks read past `dim` into zero padding); every slice in `args`
/// validated by the scorer.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
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
    accs.resize(NT * nq, 0);

    // SAFETY: preconditions above; `expand`/`block` contracts hold for every
    // token because the scorer validated `packed`, `codes`, `inv_norms`.
    unsafe {
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
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
            block::<NT>(&ws, dim, nq, qp_base, ps, sqw, &crows, &invs, accs, best);
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
            block::<1>(&one_w, dim, nq, qp_base, ps, sqw, &one_c, &one_i, accs, best);
            t += 1;
        }
    }
    best.iter().sum()
}
