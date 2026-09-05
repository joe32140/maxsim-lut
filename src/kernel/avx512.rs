//! x86_64 AVX-512 + VNNI kernel in GEMM-tile form.
//!
//! The row-per-accumulator form (one zmm per query row, 64 dims per
//! `vpdpbusd`) ends every row with a 16-lane horizontal reduction and needs a
//! sign fix-up per pair, because `vpdpbusd` is unsigned × signed. Both costs
//! disappear when the *rows* go into the lanes instead:
//!
//! * The query is stored as tiles of 16 rows × 4 consecutive plane dims
//!   (64 bytes), with every code offset by +128 so it is a valid `u8`
//!   ([`PreparedQuery::tiles_u8`]).
//! * The token's expanded weights are broadcast 4 bytes at a time.
//! * `acc = vpdpbusd(acc, q_tile, w_bcast)` then adds 4 dims for 16 rows at
//!   once; after `dim/4` steps each lane holds one row's full sum, offset by
//!   `128 · Σw`. One vector subtract per (token, tile) removes the offset
//!   exactly (integer arithmetic), and the fold runs straight on the lanes.
//!
//! Per `vpdpbusd`: one broadcast load and `1/RT` of a tile load, no
//! shuffles, no sign ops, no reductions. `RT` row tiles × `N` tokens are
//! held in registers per block (≤ 16 accumulators + `RT` tiles).
//!
//! Exactness: `Σ (q+128)·w = Σ q·w + 128·Σw` in exact integers; `|q|,|w| ≤
//! 127` keeps every partial sum far inside i32. The fold is the same op
//! sequence as every other kernel, so results are bit-identical to scalar.

use std::arch::x86_64::*;

use super::Args;
use crate::lut::{Lut, NibbleTables};
use crate::MAX_DIM;

/// Document tokens per block.
const NT: usize = 4;

/// Expand one token's `pdim` packed bytes into `kpb` planes of int8 weights,
/// returning `Σw` over all `dim` weights (the +128 offset correction).
///
/// # Safety
/// `row.len() == pdim`, `kpb · pdim <= MAX_DIM`, tables loaded for `kpb`
/// keys, `w` zero beyond `kpb · pdim`.
#[inline(always)]
unsafe fn expand(
    row: &[u8],
    tabs: &[__m128i; 8],
    nib: &NibbleTables,
    kpb: usize,
    pdim: usize,
    w: &mut [i8; MAX_DIM],
) -> i32 {
    // SAFETY: stores land in `[k·pdim, k·pdim + 16)` for `i + 16 <= pdim`;
    // the tail copies only `rem` valid lanes; the sum reads whole 64-byte
    // chunks of the 256-byte buffer, whose unused part is zero.
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
        // Σw: u8 ones × s8 w, over the 64-byte chunks covering `dim`.
        let dim = kpb * pdim;
        let ones = _mm512_set1_epi8(1);
        let mut s = _mm512_setzero_si512();
        let mut k = 0usize;
        while k < dim {
            s = _mm512_dpbusd_epi32(s, ones, _mm512_loadu_si512(wp.add(k) as *const _));
            k += 64;
        }
        _mm512_reduce_add_epi32(s)
    }
}

/// One block: `RT` row tiles × `N` tokens over all `d4n` dim groups, then the
/// fold for those rows and tokens.
///
/// # Safety
/// AVX-512F/BW/VNNI; `tiles` points at row tile `row0/16` of the query's
/// tile array with `tile_stride` bytes per tile and at least `RT` tiles
/// available; `row0 + 16·(RT−1) < nq`; `crows[j]` readable for `nq` f32s;
/// `sqw.len() == best.len() == nq`; `ws[j]` readable for `4·d4n` bytes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block<const RT: usize, const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
    corr: &[__m512i; N],
    crows: &[*const f32; N],
    invs: &[f32; N],
    tiles: *const u8,
    tile_stride: usize,
    d4n: usize,
    row0: usize,
    nq: usize,
    sqw: &[f32],
    best: &mut [f32],
) {
    // SAFETY: see function contract. Masked loads/stores keep the last,
    // partial row tile inside `sqw`/`crow`/`best`.
    unsafe {
        let zero = _mm512_setzero_si512();
        let mut acc = [[zero; RT]; N];
        for g in 0..d4n {
            let mut q = [zero; RT];
            for (rt, qt) in q.iter_mut().enumerate() {
                *qt = _mm512_loadu_si512(tiles.add(rt * tile_stride + g * 64) as *const _);
            }
            for j in 0..N {
                let wb = _mm512_set1_epi32((ws[j].as_ptr().add(g * 4) as *const i32).read_unaligned());
                for rt in 0..RT {
                    acc[j][rt] = _mm512_dpbusd_epi32(acc[j][rt], q[rt], wb);
                }
            }
        }
        for j in 0..N {
            let invv = _mm512_set1_ps(invs[j]);
            for (rt, acc_rt) in acc[j].iter().enumerate() {
                let r0 = row0 + rt * 16;
                let rem = (nq - r0).min(16);
                let mask: __mmask16 = if rem == 16 { 0xFFFF } else { (1u16 << rem) - 1 };
                let a = _mm512_cvtepi32_ps(_mm512_sub_epi32(*acc_rt, corr[j]));
                let s = _mm512_mul_ps(
                    _mm512_add_ps(
                        _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, sqw.as_ptr().add(r0)), a),
                        _mm512_maskz_loadu_ps(mask, crows[j].add(r0)),
                    ),
                    invv,
                );
                let b = _mm512_maskz_loadu_ps(mask, best.as_ptr().add(r0));
                _mm512_mask_storeu_ps(best.as_mut_ptr().add(r0), mask, _mm512_max_ps(b, s));
            }
        }
    }
}

/// All row tiles for `N` expanded tokens: blocks of 4 tiles, then the
/// remainder.
///
/// # Safety
/// As [`block`], for the whole tile array (`n16` tiles).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tokens<const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
    corr: &[__m512i; N],
    crows: &[*const f32; N],
    invs: &[f32; N],
    tiles: *const u8,
    tile_stride: usize,
    d4n: usize,
    n16: usize,
    nq: usize,
    sqw: &[f32],
    best: &mut [f32],
) {
    // SAFETY: forwards the caller's contract tile-block by tile-block.
    unsafe {
        let mut t0 = 0usize;
        while n16 - t0 >= 4 {
            let tp = tiles.add(t0 * tile_stride);
            block::<4, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                t0 * 16,
                nq,
                sqw,
                best,
            );
            t0 += 4;
        }
        let tp = tiles.add(t0 * tile_stride);
        match n16 - t0 {
            3 => block::<3, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                t0 * 16,
                nq,
                sqw,
                best,
            ),
            2 => block::<2, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                t0 * 16,
                nq,
                sqw,
                best,
            ),
            1 => block::<1, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                t0 * 16,
                nq,
                sqw,
                best,
            ),
            _ => {}
        }
    }
}

/// # Safety
/// Requires `avx512f,avx512bw,avx512vnni`; `dim % 8 == 0 && dim <= MAX_DIM`;
/// nibble tables present; every slice in `args` validated by the scorer.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
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
    let n16 = nq.div_ceil(16);
    let tile_stride = (q.stride() / 4) * 64;
    let tiles = q.tiles_u8().as_ptr();
    debug_assert!(q.tiles_u8().len() >= n16 * tile_stride);
    let sqw = q.sqw();
    best.clear();
    best.resize(nq, f32::NEG_INFINITY);

    // SAFETY: preconditions above; `expand`/`tokens` contracts hold for every
    // token because the scorer validated `packed`, `codes`, `inv_norms`, and
    // `PreparedQuery` built `n16 · tile_stride` bytes of tiles.
    unsafe {
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut corr = [_mm512_setzero_si512(); NT];
        let mut crows = [std::ptr::null::<f32>(); NT];
        let mut invs = [0f32; NT];
        let mut t = 0usize;
        while t + NT <= a.n_tokens {
            for j in 0..NT {
                let tt = t + j;
                let sumw = expand(
                    &a.packed[tt * a.row_stride..tt * a.row_stride + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                corr[j] = _mm512_set1_epi32(128 * sumw);
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
                n16,
                nq,
                sqw,
                best,
            );
            t += NT;
        }
        while t < a.n_tokens {
            let sumw = expand(
                &a.packed[t * a.row_stride..t * a.row_stride + pdim],
                &tabs,
                nib,
                kpb,
                pdim,
                &mut ws[0],
            );
            let one_w = [ws[0]];
            let one_c = [_mm512_set1_epi32(128 * sumw)];
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
                n16,
                nq,
                sqw,
                best,
            );
            t += 1;
        }
    }
    best.iter().sum()
}
