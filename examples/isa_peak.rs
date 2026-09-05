//! Sustained rate of every candidate int8 dot instruction on this core, so
//! the per-architecture kernel choice rests on a measurement rather than a
//! spec sheet.
//!
//!   cargo run --release --example isa_peak
//!
//! Each form runs 16 independent accumulator chains with its inputs reloaded
//! from a small buffer every step (so nothing can be hoisted), which gives
//! the instruction's throughput on this microarchitecture. Numbers are
//! printed as G MAC/s (int8 multiply-accumulates per nanosecond) so the
//! forms are comparable: `sdot` does 16 MACs, `smmla` 32, a
//! `vpmaddubsw`+`vpmaddwd` pair 32, `vpdpbusd` 32 on ymm and 64 on zmm.
//!
//! On aarch64 two extra rows run the MaxSim block's actual load pattern for
//! the `sdot` and the `smmla` register blocking, because `smmla` needs the
//! two tokens' weights interleaved and that costs instructions too.
//!
//! What the numbers decide: a core where `smmla` matches `sdot` in
//! instructions per cycle does twice the MACs with it, and the `neon-i8mm`
//! kernel wins there; a core that issues `smmla` at half the rate (Apple
//! M-series) gains nothing from it. Likewise on x86 a single 256-bit
//! `vpdpbusd` replaces the three-instruction AVX2 sequence wherever AVX-VNNI
//! exists.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod common {
    use std::time::Instant;

    pub const ITERS: usize = 20_000_000;
    pub const CHAINS: usize = 16;

    /// Best of three runs of a timed section returning (seconds, checksum).
    pub fn best<F: FnMut() -> (f64, i32)>(mut f: F) -> (f64, i32) {
        let mut b = (f64::INFINITY, 0);
        for _ in 0..3 {
            let r = f();
            if r.0 < b.0 {
                b = r;
            }
        }
        b
    }

    pub fn report(form: &str, macs_per_instr: usize, instrs: usize, (secs, checksum): (f64, i32)) {
        let gi = instrs as f64 / secs / 1e9;
        println!(
            "{form:>42}: {gi:6.2} G instr/s  {:7.1} G MAC/s   (checksum {checksum})",
            gi * macs_per_instr as f64
        );
    }

    pub fn timed<F: FnOnce() -> i32>(f: F) -> (f64, i32) {
        let t = Instant::now();
        let sum = f();
        (t.elapsed().as_secs_f64(), sum)
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use super::common::*;
    use std::arch::aarch64::*;

    /// `sdot`: acc.4s += Σ₄ a.16b × b.16b per lane (16 MACs).
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let out: int32x4_t;
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

    /// `smmla`: acc (2×2 i32) += A (2×8 i8) · Bᵀ (8×2 i8), 32 MACs. The
    /// result lanes are `[a0·b0, a0·b1, a1·b0, a1·b1]`.
    #[inline(always)]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let out: int32x4_t;
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

    macro_rules! chain_fn {
        ($name:ident, $op:ident, $feat:literal) => {
            #[target_feature(enable = $feat)]
            unsafe fn $name(buf: &[i8]) -> (f64, i32) {
                unsafe {
                    let p = buf.as_ptr();
                    let mut acc = [vdupq_n_s32(0); CHAINS];
                    timed(|| {
                        for i in 0..ITERS {
                            let off = (i & 63) * 16;
                            let a = vld1q_s8(p.add(off));
                            let b = vld1q_s8(p.add(off + 1024));
                            for c in acc.iter_mut() {
                                *c = $op(*c, a, b);
                            }
                        }
                        acc.iter().fold(0i32, |s, &v| s.wrapping_add(vaddvq_s32(v)))
                    })
                }
            }
        };
    }
    chain_fn!(sdot_chains, sdot, "dotprod");
    chain_fn!(smmla_chains, smmla, "i8mm");

    /// The `neon-sdot` block: 4 query rows × 4 tokens × 16 dims per step,
    /// 8 loads and 16 `sdot`.
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot_block(buf: &[i8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let mut acc = [vdupq_n_s32(0); 16];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 16;
                    let q = [
                        vld1q_s8(p.add(off)),
                        vld1q_s8(p.add(off + 1024)),
                        vld1q_s8(p.add(off + 2048)),
                        vld1q_s8(p.add(off + 3072)),
                    ];
                    for j in 0..4 {
                        let w = vld1q_s8(p.add(off + 4096 + j * 1024));
                        for r in 0..4 {
                            acc[j * 4 + r] = sdot(acc[j * 4 + r], q[r], w);
                        }
                    }
                }
                acc.iter().fold(0i32, |s, &v| s.wrapping_add(vaddvq_s32(v)))
            })
        }
    }

    /// The `neon-i8mm` block: 8 query rows (4 row pairs, pre-interleaved)
    /// × 4 tokens (2 pairs, interleaved on the fly with 4 `vcombine`) × 16
    /// dims per step: 12 loads, 4 interleaves, 16 `smmla` = 512 MACs.
    #[target_feature(enable = "i8mm")]
    unsafe fn smmla_block(buf: &[i8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let mut acc = [vdupq_n_s32(0); 8];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 32;
                    let q: [int8x16_t; 8] =
                        std::array::from_fn(|k| vld1q_s8(p.add(off + (k / 2) * 1024 + (k % 2) * 16)));
                    let w0 = vld1q_s8(p.add(off + 4096));
                    let w1 = vld1q_s8(p.add(off + 5120));
                    let w2 = vld1q_s8(p.add(off + 6144));
                    let w3 = vld1q_s8(p.add(off + 7168));
                    let wa_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
                    let wa_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
                    let wb_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
                    let wb_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));
                    for rp in 0..4 {
                        acc[rp] = smmla(smmla(acc[rp], q[rp * 2], wa_lo), q[rp * 2 + 1], wa_hi);
                        acc[4 + rp] = smmla(smmla(acc[4 + rp], q[rp * 2], wb_lo), q[rp * 2 + 1], wb_hi);
                    }
                }
                acc.iter().fold(0i32, |s, &v| s.wrapping_add(vaddvq_s32(v)))
            })
        }
    }

    pub fn run() {
        let buf: Vec<i8> = (0..8192).map(|i| (i % 7) as i8 - 3).collect();
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        let i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        println!("features: dotprod {dotprod}, i8mm {i8mm}");
        let n = CHAINS * ITERS;
        if dotprod {
            // SAFETY: feature detected.
            report("sdot, 16 chains", 16, n, best(|| unsafe { sdot_chains(&buf) }));
        }
        if i8mm {
            // SAFETY: feature detected.
            report("smmla, 16 chains", 32, n, best(|| unsafe { smmla_chains(&buf) }));
        }
        if dotprod {
            // SAFETY: feature detected.
            report(
                "sdot block 4 rows x 4 tok (16 sdot)",
                16,
                16 * ITERS,
                best(|| unsafe { sdot_block(&buf) }),
            );
        }
        if i8mm {
            // SAFETY: feature detected.
            report(
                "smmla block 8 rows x 4 tok (16 smmla)",
                32,
                16 * ITERS,
                best(|| unsafe { smmla_block(&buf) }),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use super::common::*;
    use std::arch::x86_64::*;

    /// The AVX2 sequence: `vpmaddubsw` (u8×s8 → i16 pairs), `vpmaddwd` by
    /// ones (→ i32), `vpaddd`. Three instructions for 32 MACs.
    #[target_feature(enable = "avx2")]
    unsafe fn maddubs_chains(buf: &[u8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let ones = _mm256_set1_epi16(1);
            let mut acc = [_mm256_setzero_si256(); CHAINS];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 32;
                    let a = _mm256_loadu_si256(p.add(off) as *const __m256i);
                    let b = _mm256_loadu_si256(p.add(off + 2048) as *const __m256i);
                    for c in acc.iter_mut() {
                        *c = _mm256_add_epi32(*c, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
                    }
                }
                acc.iter().fold(0i32, |s, &v| s.wrapping_add(hsum256(v)))
            })
        }
    }

    /// 256-bit `vpdpbusd` via the AVX-VNNI encoding (no AVX-512 needed).
    #[target_feature(enable = "avxvnni")]
    unsafe fn avxvnni_chains(buf: &[u8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let mut acc = [_mm256_setzero_si256(); CHAINS];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 32;
                    let a = _mm256_loadu_si256(p.add(off) as *const __m256i);
                    let b = _mm256_loadu_si256(p.add(off + 2048) as *const __m256i);
                    for c in acc.iter_mut() {
                        *c = _mm256_dpbusd_avx_epi32(*c, a, b);
                    }
                }
                acc.iter().fold(0i32, |s, &v| s.wrapping_add(hsum256(v)))
            })
        }
    }

    /// 256-bit `vpdpbusd` via the AVX-512VL encoding.
    #[target_feature(enable = "avx512vl,avx512vnni")]
    unsafe fn vl_chains(buf: &[u8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let mut acc = [_mm256_setzero_si256(); CHAINS];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 32;
                    let a = _mm256_loadu_si256(p.add(off) as *const __m256i);
                    let b = _mm256_loadu_si256(p.add(off + 2048) as *const __m256i);
                    for c in acc.iter_mut() {
                        *c = _mm256_dpbusd_epi32(*c, a, b);
                    }
                }
                acc.iter().fold(0i32, |s, &v| s.wrapping_add(hsum256(v)))
            })
        }
    }

    /// 512-bit `vpdpbusd`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn zmm_chains(buf: &[u8]) -> (f64, i32) {
        unsafe {
            let p = buf.as_ptr();
            let mut acc = [_mm512_setzero_si512(); CHAINS];
            timed(|| {
                for i in 0..ITERS {
                    let off = (i & 63) * 64;
                    let a = _mm512_loadu_si512(p.add(off) as *const _);
                    let b = _mm512_loadu_si512(p.add(off + 4096) as *const _);
                    for c in acc.iter_mut() {
                        *c = _mm512_dpbusd_epi32(*c, a, b);
                    }
                }
                acc.iter()
                    .fold(0i32, |s, &v| s.wrapping_add(_mm512_reduce_add_epi32(v)))
            })
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256i) -> i32 {
        let mut tmp = [0i32; 8];
        unsafe { _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v) };
        tmp.iter().fold(0i32, |s, &x| s.wrapping_add(x))
    }

    pub fn run() {
        let buf: Vec<u8> = (0..8192).map(|i| (i % 7) as u8 + 1).collect();
        let avx2 = is_x86_feature_detected!("avx2");
        let avxvnni = is_x86_feature_detected!("avxvnni");
        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni");
        let vl = avx512 && is_x86_feature_detected!("avx512vl");
        println!("features: avx2 {avx2}, avxvnni {avxvnni}, avx512 f/bw/vnni {avx512}, avx512vl {vl}");
        let n = CHAINS * ITERS;
        if avx2 {
            // SAFETY: feature detected.
            report(
                "vpmaddubsw+vpmaddwd+vpaddd ymm (AVX2)",
                32,
                n,
                best(|| unsafe { maddubs_chains(&buf) }),
            );
        }
        if avxvnni {
            // SAFETY: feature detected.
            report(
                "vpdpbusd ymm (AVX-VNNI)",
                32,
                n,
                best(|| unsafe { avxvnni_chains(&buf) }),
            );
        }
        if vl {
            // SAFETY: feature detected.
            report(
                "vpdpbusd ymm (AVX-512VL)",
                32,
                n,
                best(|| unsafe { vl_chains(&buf) }),
            );
        }
        if avx512 {
            // SAFETY: feature detected.
            report(
                "vpdpbusd zmm (AVX-512)",
                64,
                n,
                best(|| unsafe { zmm_chains(&buf) }),
            );
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod arch {
    pub fn run() {
        println!("no SIMD int8 dot instruction to measure on this architecture");
    }
}

fn main() {
    println!("arch {}", std::env::consts::ARCH);
    arch::run();
}
