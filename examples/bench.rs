//! Nanoseconds per scored document token, dispatched kernel vs the scalar
//! reference, on synthetic ColBERT-shaped data. Prints the kernel that
//! actually ran.
//!
//!   cargo run --release --example bench -- [dim] [nbits] [nq] [doc_tokens] [n_docs]
//!
//! On an Apple Silicon machine whose rustup default is x86_64, pass
//! `--target aarch64-apple-darwin` or you will benchmark Rosetta.

use std::time::Instant;

use maxsim_lut::{Codes, ColbertPacking, DocView, Lut, Packing, PreparedQuery, Scorer};

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
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let p = ColbertPacking::new(nbits).unwrap();
    let nb = 1usize << nbits;
    let mut w: Vec<f32> = (0..nb).map(|_| rng.f32(-0.4, 0.4)).collect();
    w.sort_by(|a, b| a.total_cmp(b));
    let lut = Lut::new(&p, &w).unwrap();
    let lut_scalar = lut.clone().force_scalar(true);

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

    println!(
        "dim {dim}, nbits {nbits}, {nq} query tokens, {ndocs} docs × {ntok} tokens, {ncent} centroids\narch {}, kernel: {}",
        std::env::consts::ARCH,
        lut.kernel(dim)
    );

    let run = |label: &str, lut: &Lut| {
        let s = Scorer::new(lut, &q).with_centroid_term(&cdot, ncent).unwrap();
        let mut out = vec![0.0f32; ndocs];
        s.score_many(docs.iter().copied(), &mut out); // warm
        let reps = 5;
        let mut best_ns = f64::INFINITY;
        let mut checksum = 0.0f64;
        for _ in 0..reps {
            let t = Instant::now();
            s.score_many(docs.iter().copied(), &mut out);
            let ns = t.elapsed().as_nanos() as f64 / (ndocs * ntok) as f64;
            best_ns = best_ns.min(ns);
            checksum = out.iter().map(|&v| v as f64).sum();
        }
        println!(
            "{label:>28}: {best_ns:7.2} ns/token   ({:.1} µs/doc, checksum {checksum:.3})",
            best_ns * ntok as f64 / 1e3
        );
        (best_ns, checksum)
    };
    let (fast, c1) = run(&format!("{}", lut.kernel(dim)), &lut);
    let (slow, c2) = run("scalar reference", &lut_scalar);
    assert_eq!(c1.to_bits(), c2.to_bits(), "kernels disagree");
    println!("{:>28}: {:.2}x", "speedup", slow / fast);
}
