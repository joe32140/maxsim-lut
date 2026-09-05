//! Nanoseconds per scored document token for **every kernel this CPU can
//! run**, plus the scalar reference, on synthetic ColBERT-shaped data.
//!
//!   cargo run --release --example bench -- [dim] [nbits] [nq] [doc_tokens] [n_docs]
//!
//! The arms are interleaved round by round inside one process, and each is
//! reduced by its minimum, so a scheduling hiccup has to hit every round of
//! one arm to change the ranking. Comparing separate `cargo run` invocations
//! instead is how a busy machine invents a regression: on a hybrid CPU one
//! run can land on an efficiency core and read 2× slow.
//!
//! The spread between an arm's minimum and its median is printed as a noise
//! verdict. Above about 10% the machine is too busy for the small
//! differences between kernels to mean anything.
//!
//! On an Apple Silicon machine whose rustup default is x86_64, pass
//! `--target aarch64-apple-darwin` or you will benchmark Rosetta.

use std::time::Instant;

use maxsim_lut::{supported_kernels, Codes, ColbertPacking, DocView, Lut, Packing, PreparedQuery, Scorer};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

/// One measured arm: a label, the table configured for it, and its timings.
struct Arm {
    label: String,
    lut: Lut,
    ns: Vec<f64>,
    checksum: f64,
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer arg"))
        .collect();
    let dim = args.first().copied().unwrap_or(128);
    let nbits = args.get(1).copied().unwrap_or(4);
    let nq = args.get(2).copied().unwrap_or(32);
    let ntok = args.get(3).copied().unwrap_or(240);
    let ndocs = args.get(4).copied().unwrap_or(1024);
    let ncent = 16_384;
    let reps = 9;
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let p = ColbertPacking::new(nbits).unwrap();
    let nb = 1usize << nbits;
    let mut w: Vec<f32> = (0..nb).map(|_| rng.f32(-0.4, 0.4)).collect();
    w.sort_by(|a, b| a.total_cmp(b));
    let lut = Lut::new(&p, &w).unwrap();

    let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
    let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
    let cdot: Vec<f32> = (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect();

    let pdim = dim / p.keys_per_byte();
    let mut packed = vec![0u8; ndocs * ntok * pdim];
    for b in packed.iter_mut() {
        *b = (rng.next() >> 56) as u8;
    }
    let codes: Vec<u32> = (0..ndocs * ntok)
        .map(|_| (rng.next() % ncent as u64) as u32)
        .collect();
    let inv: Vec<f32> = (0..ndocs * ntok).map(|_| rng.f32(0.8, 1.2)).collect();
    let docs: Vec<DocView> = (0..ndocs)
        .map(|d| {
            DocView::new(&packed[d * ntok * pdim..(d + 1) * ntok * pdim], ntok, pdim)
                .codes(Codes::U32(&codes[d * ntok..(d + 1) * ntok]))
                .inv_norms(&inv[d * ntok..(d + 1) * ntok])
        })
        .collect();

    // One arm per executable kernel, plus the scalar reference. `dispatch`
    // is what an unpinned host would get, named so the calibrated choice is
    // visible next to the kernels it chose between.
    let dispatched = lut.kernel(dim);
    let mut arms: Vec<Arm> = Vec::new();
    for &k in supported_kernels() {
        arms.push(Arm {
            label: if k == dispatched {
                format!("{k} (dispatched)")
            } else {
                format!("{k}")
            },
            lut: lut.clone().pin_kernel(Some(k)),
            ns: Vec::new(),
            checksum: 0.0,
        });
    }
    if !dispatched.is_simd() {
        arms.push(Arm {
            label: format!("{dispatched} (dispatched)"),
            lut: lut.clone(),
            ns: Vec::new(),
            checksum: 0.0,
        });
    }
    arms.push(Arm {
        label: "scalar reference".to_string(),
        lut: lut.clone().force_scalar(true),
        ns: Vec::new(),
        checksum: 0.0,
    });

    println!(
        "dim {dim}, nbits {nbits}, {nq} query tokens, {ndocs} docs × {ntok} tokens, {ncent} centroids\narch {}, dispatched kernel: {dispatched}, {reps} interleaved rounds",
        std::env::consts::ARCH,
    );

    let mut out = vec![0.0f32; ndocs];
    for arm in arms.iter_mut() {
        let s = Scorer::new(&arm.lut, &q)
            .with_centroid_term(&cdot, ncent)
            .unwrap();
        s.score_many(docs.iter().copied(), &mut out); // warm caches and branch predictors
    }
    for _ in 0..reps {
        for arm in arms.iter_mut() {
            let s = Scorer::new(&arm.lut, &q)
                .with_centroid_term(&cdot, ncent)
                .unwrap();
            let t = Instant::now();
            s.score_many(docs.iter().copied(), &mut out);
            arm.ns.push(t.elapsed().as_nanos() as f64 / (ndocs * ntok) as f64);
            arm.checksum = out.iter().map(|&v| v as f64).sum();
        }
    }

    let reference = arms.last().expect("at least the scalar arm");
    let (slow, want) = {
        let mut v = reference.ns.clone();
        v.sort_by(f64::total_cmp);
        (v[0], reference.checksum)
    };
    let mut worst_spread = 0.0f64;
    for arm in &arms {
        let mut v = arm.ns.clone();
        v.sort_by(f64::total_cmp);
        let (best, med) = (v[0], median(&v));
        let spread = (med - best) / best;
        worst_spread = worst_spread.max(spread);
        assert_eq!(
            arm.checksum.to_bits(),
            want.to_bits(),
            "{}: checksum {} differs from the scalar reference {want}",
            arm.label,
            arm.checksum
        );
        println!(
            "{:>26}: {best:7.2} ns/token  (median {med:7.2}, {:5.1} µs/doc, {:5.2}x scalar)",
            arm.label,
            best * ntok as f64 / 1e3,
            slow / best,
        );
    }
    println!(
        "{:>26}: {:.1}% median-vs-best spread — {}",
        "noise",
        worst_spread * 100.0,
        if worst_spread < 0.10 {
            "quiet enough to compare kernels"
        } else {
            "TOO NOISY, differences under ~2x are not real; free the machine and rerun"
        }
    );
}
