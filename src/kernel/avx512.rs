//! x86_64 AVX-512 + VNNI kernel.
//!
//! The expand stays 128-bit `pshufb` (charged once per doc token and
//! amortised over every query row); what moves to 512 bits is the part
//! charged per (query row, token) plus the fold. The dot uses `vpdpbusd`
//! (unsigned × signed): we feed it `|w|` against `sign(w)·q`. There is no
//! 512-bit `vpsignb`, so the sign is applied with a mask: `movepi8_mask`
//! extracts w's sign bits and `mask_sub_epi8` negates exactly those query
//! lanes. `-128` never occurs on either side (both clamped to ±127), so the
//! negation is exact, and `w == 0` lanes zero the product regardless.

use std::arch::x86_64::*;

use super::Args;
use crate::lut::Lut;
use crate::MAX_DIM;

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
    accs.resize(nq, 0);
    let mut w = [0i8; MAX_DIM];

    // SAFETY: pointer arithmetic stays within validated slices, the
    // `[i8; MAX_DIM]` buffer (64-lane chunks below a multiple-of-64 bound
    // <= 256) and the zero-padded query planes (stride multiple of 64).
    unsafe {
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }
        let low_mask = _mm_set1_epi8(0x0F);
        let zero = _mm512_setzero_si512();

        for t in 0..a.n_tokens {
            let row = &a.packed[t * a.row_stride..t * a.row_stride + pdim];
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
            let wp = w.as_ptr();
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = qp_base.add(qi * ps);
                let mut acc = zero;
                let mut k = 0usize;
                while k < dim {
                    let qv = _mm512_loadu_si512(qp.add(k) as *const _);
                    let wv = _mm512_loadu_si512(wp.add(k) as *const _);
                    let mag = _mm512_abs_epi8(wv);
                    let neg = _mm512_movepi8_mask(wv);
                    let sq = _mm512_mask_sub_epi8(qv, neg, zero, qv);
                    acc = _mm512_dpbusd_epi32(acc, mag, sq);
                    k += 64;
                }
                *acc_qi = _mm512_reduce_add_epi32(acc);
            }
            fold_block(accs, sqw, a.crow(t), a.inv(t), best);
        }
    }
    best.iter().sum()
}
