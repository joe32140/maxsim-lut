//! x86_64 AVX2 kernel in GEMM-tile form (see `avx512.rs` for the idea).
//!
//! Rows go into the lanes: a 32-byte load of the query tile is 8 rows × 4
//! consecutive plane dims, multiplied against a 4-byte weight broadcast. The
//! only AVX2 `u8 × s8` dot is `vpmaddubsw`, whose pair sums saturate at
//! i16, so the +128 offset trick of the AVX-512 path is not available
//! (`(q+128)·w` pairs overflow). Instead the unsigned operand is `|q|` and
//! the sign of `q` is moved onto the broadcast weights with `vpsignb`:
//! `|q| · sign(q)·w = q·w`, exact, and `|q|,|w| ≤ 127` keeps each pair sum
//! under 32 767. The signed tile is recovered from the stored `u8` tile by
//! `xor 0x80` (two's complement of `q+128`), so no second query buffer.
//!
//! Per (token, 8 rows, 4 dims): one broadcast load, `vpsignb`, `vpmaddubsw`,
//! `vpmaddwd`, `vpaddd`. No horizontal reductions: the fold runs on lanes.
//! Compared with the row-per-accumulator form, this removes the 8-lane
//! reduction that was paid per (row, token).

use std::arch::x86_64::*;

use super::Args;
use crate::lut::{Lut, NibbleTables};
use crate::MAX_DIM;

/// Document tokens per block.
const NT: usize = 4;

/// Expand one token's `pdim` packed bytes into `kpb` planes of int8 weights.
///
/// # Safety
/// `row.len() == pdim`, `kpb · pdim <= MAX_DIM`, tables loaded for `kpb` keys.
#[inline(always)]
pub(super) unsafe fn expand(
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

/// Lane mask for the first `rem` of 8 f32 lanes (all-ones = load/store).
///
/// # Safety
/// AVX2.
#[inline(always)]
pub(super) unsafe fn lane_mask(rem: usize) -> __m256i {
    // SAFETY: register-only ops.
    unsafe {
        let idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _mm256_cmpgt_epi32(_mm256_set1_epi32(rem as i32), idx)
    }
}

/// One block: `RT` eight-row tiles × `N` tokens over all `d4n` dim groups,
/// then the fold for those rows and tokens.
///
/// # Safety
/// AVX2; `tiles` points at 8-row tile `row0/8` inside the query's `u8` tile
/// array (16-row tiles of `tile_stride` bytes; 8-row tile `h` of 16-row
/// tile `t` is at `t·tile_stride + 32·(h % 2)`), with `RT` such tiles
/// available and `row0 + 8·(RT−1) < nq`; `crows[j]` readable for `nq` f32s;
/// `sqw.len() == best.len() == nq`; `ws[j]` readable for `4·d4n` bytes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block<const RT: usize, const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
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
        let ones = _mm256_set1_epi16(1);
        let flip = _mm256_set1_epi8(-128);
        let mut acc = [[zero; RT]; N];
        for g in 0..d4n {
            let mut qs = [zero; RT];
            let mut qa = [zero; RT];
            for rt in 0..RT {
                let qu = _mm256_loadu_si256(tile_ptrs[rt].add(g * 64) as *const __m256i);
                qs[rt] = _mm256_xor_si256(qu, flip);
                qa[rt] = _mm256_abs_epi8(qs[rt]);
            }
            for j in 0..N {
                let wb = _mm256_set1_epi32((ws[j].as_ptr().add(g * 4) as *const i32).read_unaligned());
                for rt in 0..RT {
                    let prod = _mm256_maddubs_epi16(qa[rt], _mm256_sign_epi8(wb, qs[rt]));
                    acc[j][rt] = _mm256_add_epi32(acc[j][rt], _mm256_madd_epi16(prod, ones));
                }
            }
        }
        for j in 0..N {
            let invv = _mm256_set1_ps(invs[j]);
            for (rt, acc_rt) in acc[j].iter().enumerate() {
                let r0 = row0 + rt * 8;
                let rem = (nq - r0).min(8);
                let m = lane_mask(rem);
                let a = _mm256_cvtepi32_ps(*acc_rt);
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

/// Address of 8-row tile `r8` in the `u8` tile array.
#[inline(always)]
pub(super) unsafe fn tile8(tiles: *const u8, tile_stride: usize, r8: usize) -> *const u8 {
    // SAFETY: caller guarantees `r8 < 2 · n16`.
    unsafe { tiles.add((r8 / 2) * tile_stride + (r8 % 2) * 32) }
}

/// All 8-row tiles for `N` expanded tokens: pairs of tiles, then a remainder.
///
/// # Safety
/// As [`block`], for the whole tile array (`n8 = ⌈nq/8⌉` tiles).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tokens<const N: usize>(
    ws: &[[i8; MAX_DIM]; N],
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
            block::<2, N>(ws, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
            t0 += 2;
        }
        if n8 - t0 == 1 {
            let tp = [tile8(tiles, tile_stride, t0)];
            block::<1, N>(ws, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
        }
    }
}

/// # Safety
/// Requires AVX2; `dim % 8 == 0 && dim <= MAX_DIM`; nibble tables present;
/// every slice in `args` validated by the scorer.
#[target_feature(enable = "avx2")]
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
            tokens::<NT>(&ws, &crows, &invs, tiles, tile_stride, d4n, n8, nq, sqw, best);
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
            let one_r = [a.crow(t)];
            let one_i = [a.inv(t)];
            tokens::<1>(&one_w, &one_r, &one_i, tiles, tile_stride, d4n, n8, nq, sqw, best);
            t += 1;
        }
    }
    best.iter().sum()
}
