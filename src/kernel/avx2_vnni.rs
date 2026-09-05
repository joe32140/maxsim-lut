//! x86_64 AVX-VNNI kernel: the AVX2 GEMM-tile shape with a real `u8 × s8`
//! dot instruction.
//!
//! `vpdpbusd` accumulates four `u8 × s8` products straight into each i32
//! lane. AVX-512 machines get it through the AVX-512VL encoding; since
//! Alder Lake and Zen 5 it also exists as a VEX-encoded 256-bit instruction
//! (`avxvnni`) on parts that have **no** AVX-512 at all. That is the gap
//! this kernel fills: those cores otherwise fall back to the three-instruction
//! `vpmaddubsw`/`vpmaddwd`/`vpaddd` sequence of [`super::avx2`].
//!
//! Everything else is [`super::avx2`]: the same `u8` query tiles (8 rows × 4
//! plane dims per 32-byte load), the same weight broadcast, the same masked
//! fold. Two differences follow from `vpdpbusd` being non-saturating:
//!
//! * the query is used in its stored `+128` form directly, with no
//!   `vpsignb`/`vpabsb` fix-up, and
//! * the resulting `128 · Σw` offset is subtracted once per (token, tile)
//!   in exact integer arithmetic, as in [`super::avx512`].
//!
//! Three instructions per 32 MACs become one, and the query no longer needs
//! two derived registers per tile.

use std::arch::x86_64::*;

use super::avx2::{expand, lane_mask, tile8};
use super::Args;
use crate::lut::Lut;
use crate::MAX_DIM;

/// Document tokens per block.
const NT: usize = 4;

/// `Σw` over the token's `dim` expanded weights, as `u8 ones × s8 w`.
///
/// # Safety
/// AVX-VNNI; `w` zero beyond `dim`; `dim <= MAX_DIM`.
#[inline(always)]
unsafe fn sum_weights(w: &[i8; MAX_DIM], dim: usize) -> i32 {
    // SAFETY: reads whole 32-byte chunks of the 256-byte buffer, whose
    // unused tail is zero and contributes nothing to the sum.
    unsafe {
        let ones = _mm256_set1_epi8(1);
        let mut s = _mm256_setzero_si256();
        let mut k = 0usize;
        while k < dim {
            let v = _mm256_loadu_si256(w.as_ptr().add(k) as *const __m256i);
            s = _mm256_dpbusd_avx_epi32(s, ones, v);
            k += 32;
        }
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256(s, 1);
        let mut tmp = [0i32; 4];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, _mm_add_epi32(lo, hi));
        tmp.iter().sum()
    }
}

/// One block: `RT` eight-row tiles × `N` tokens over all `d4n` dim groups,
/// then the fold for those rows and tokens.
///
/// # Safety
/// AVX-VNNI; `tile_ptrs[rt]` points at 8-row tile `row0/8 + rt` of the
/// query's tile array; `row0 + 8·(RT−1) < nq`; `crows[j]` readable for `nq`
/// f32s; `sqw.len() == best.len() == nq`; `ws[j]` readable for `4·d4n` bytes
/// and `corr[j] == 128 · Σw` for that token.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block<const RT: usize, const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
    corr: &[__m256i; N],
    crows: &[*const f32; N],
    invs: &[f32; N],
    tile_ptrs: &[*const u8; RT],
    d4n: usize,
    row0: usize,
    nq: usize,
    sqw: &[f32],
    best: &mut [f32],
) {
    // SAFETY: see function contract. Masked loads/stores keep the last,
    // partial row tile inside `sqw`/`crow`/`best`.
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut acc = [[zero; RT]; N];
        for g in 0..d4n {
            let mut q = [zero; RT];
            for (rt, qv) in q.iter_mut().enumerate() {
                *qv = _mm256_loadu_si256(tile_ptrs[rt].add(g * 64) as *const __m256i);
            }
            for j in 0..N {
                let wb = _mm256_set1_epi32((ws[j].as_ptr().add(g * 4) as *const i32).read_unaligned());
                for rt in 0..RT {
                    acc[j][rt] = _mm256_dpbusd_avx_epi32(acc[j][rt], q[rt], wb);
                }
            }
        }
        for j in 0..N {
            let invv = _mm256_set1_ps(invs[j]);
            for (rt, acc_rt) in acc[j].iter().enumerate() {
                let r0 = row0 + rt * 8;
                let rem = (nq - r0).min(8);
                let m = lane_mask(rem);
                let a = _mm256_cvtepi32_ps(_mm256_sub_epi32(*acc_rt, corr[j]));
                let s = _mm256_mul_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(_mm256_maskload_ps(sqw.as_ptr().add(r0), m), a),
                        _mm256_maskload_ps(crows[j].add(r0), m),
                    ),
                    invv,
                );
                let b = _mm256_maskload_ps(best.as_ptr().add(r0), m);
                _mm256_maskstore_ps(best.as_mut_ptr().add(r0), m, _mm256_max_ps(b, s));
            }
        }
    }
}

/// All 8-row tiles for `N` expanded tokens: pairs of tiles, then a remainder.
///
/// # Safety
/// As [`block`], for the whole tile array (`n8 = ⌈nq/8⌉` tiles).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tokens<const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
    corr: &[__m256i; N],
    crows: &[*const f32; N],
    invs: &[f32; N],
    tiles: *const u8,
    tile_stride: usize,
    d4n: usize,
    n8: usize,
    nq: usize,
    sqw: &[f32],
    best: &mut [f32],
) {
    // SAFETY: forwards the caller's contract tile-block by tile-block.
    unsafe {
        let mut t0 = 0usize;
        while n8 - t0 >= 2 {
            let tp = [tile8(tiles, tile_stride, t0), tile8(tiles, tile_stride, t0 + 1)];
            block::<2, N>(ws, corr, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
            t0 += 2;
        }
        if n8 - t0 == 1 {
            let tp = [tile8(tiles, tile_stride, t0)];
            block::<1, N>(ws, corr, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
        }
    }
}

/// # Safety
/// Requires `avx2,avxvnni`; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables
/// present; every slice in `args` validated by the scorer.
#[target_feature(enable = "avx2,avxvnni")]
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
    let d4n = dim / 4;
    let n8 = nq.div_ceil(8);
    let tile_stride = (q.stride() / 4) * 64;
    let tiles = q.tiles_u8().as_ptr();
    debug_assert!(q.tiles_u8().len() >= nq.div_ceil(16) * tile_stride);
    let sqw = q.sqw();
    best.clear();
    best.resize(nq, f32::NEG_INFINITY);

    // SAFETY: preconditions above; `expand`/`tokens` contracts hold for every
    // token because the scorer validated `packed`, `codes`, `inv_norms`, and
    // `PreparedQuery` built `⌈nq/16⌉ · tile_stride` bytes of tiles.
    unsafe {
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut corr = [_mm256_setzero_si256(); NT];
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
                corr[j] = _mm256_set1_epi32(128 * sum_weights(&ws[j], dim));
                crows[j] = a.crow(tt);
                invs[j] = a.inv(tt);
            }
            tokens::<NT>(
                &ws,
                &corr,
                &crows,
                &invs,
                tiles,
                tile_stride,
                d4n,
                n8,
                nq,
                sqw,
                best,
            );
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
            let one_c = [_mm256_set1_epi32(128 * sum_weights(&ws[0], dim))];
            let one_r = [a.crow(t)];
            let one_i = [a.inv(t)];
            tokens::<1>(
                &one_w,
                &one_c,
                &one_r,
                &one_i,
                tiles,
                tile_stride,
                d4n,
                n8,
                nq,
                sqw,
                best,
            );
            t += 1;
        }
    }
    best.iter().sum()
}
