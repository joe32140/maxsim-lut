//! x86_64 AVX2 kernel: `pshufb` nibble expansion into plane order,
//! `maddubs`/`madd` accumulation, eight-row vectorised epilogue.
//!
//! Exactness: both operands are clamped to ±127 at quantisation, so
//! `_mm256_sign_epi8` never sees −128 and each `maddubs` pair-sum is bounded
//! by 2·127·127 < i16::MAX; the i32 accumulator is exact.

use std::arch::x86_64::*;

use super::Args;
use crate::lut::Lut;
use crate::MAX_DIM;

/// Eight-row fold; same ops, same order as the scalar tail, hence bit-identical
/// (`_mm256_cvtepi32_ps` rounds to nearest like `as f32`; separate mul/add;
/// `inv` last; `_mm256_max_ps` matches the scalar select for finite scores).
///
/// # Safety
/// `crow` readable for `accs.len()` f32s.
#[inline(always)]
unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
    let nq = accs.len();
    // SAFETY: see function contract; vector loads stay within `i + 8 <= nq`.
    unsafe {
        let invv = _mm256_set1_ps(inv);
        let mut i = 0usize;
        while i + 8 <= nq {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(accs.as_ptr().add(i) as *const __m256i));
            let s = _mm256_mul_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(sqw.as_ptr().add(i)), a),
                    _mm256_loadu_ps(crow.add(i)),
                ),
                invv,
            );
            let b = _mm256_loadu_ps(best.as_ptr().add(i));
            _mm256_storeu_ps(best.as_mut_ptr().add(i), _mm256_max_ps(b, s));
            i += 8;
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
/// Requires AVX2; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables present;
/// query plane stride `>= padded_stride(dim)` (32-byte chunks read past
/// `dim` into zero padding); every slice in `args` validated by the scorer.
#[target_feature(enable = "avx2")]
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
    // `[i8; MAX_DIM]` buffer (32-lane chunks below a multiple-of-32 bound
    // <= 256) and the zero-padded query planes (stride multiple of 64).
    unsafe {
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }
        let low_mask = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);

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
                let mut acc = _mm256_setzero_si256();
                let mut k = 0usize;
                while k < dim {
                    let qv = _mm256_loadu_si256(qp.add(k) as *const __m256i);
                    let wv = _mm256_loadu_si256(wp.add(k) as *const __m256i);
                    let prod = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(qv, wv));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
                    k += 32;
                }
                let hi128 = _mm256_extracti128_si256(acc, 1);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
                *acc_qi = _mm_cvtsi128_si32(s32);
            }
            fold_block(accs, sqw, a.crow(t), a.inv(t), best);
        }
    }
    best.iter().sum()
}
