//! The contract: every path returns the same bits as the scalar reference,
//! the reference matches an f64 recomputation, and the optional terms behave
//! as identities.

use maxsim_lut::{Codes, ColbertPacking, DocView, Kernel, Lut, Packing, PreparedQuery, Scorer};

/// Deterministic xorshift so the tests carry no `rand` dependency.
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
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn weights(nbits: usize, rng: &mut Rng) -> Vec<f32> {
    // Sorted quantile-like weights in [-0.4, 0.4], as a residual quantiser produces.
    let n = 1usize << nbits;
    let mut w: Vec<f32> = (0..n).map(|_| rng.f32(-0.4, 0.4)).collect();
    w.sort_by(|a, b| a.total_cmp(b));
    w
}

struct Doc {
    packed: Vec<u8>,
    buckets: Vec<usize>, // [ntok * dim]
    codes: Vec<u32>,
    inv: Vec<f32>,
    ntok: usize,
    stride: usize,
}

fn make_doc(
    p: &ColbertPacking,
    dim: usize,
    ntok: usize,
    ncent: usize,
    extra_stride: usize,
    rng: &mut Rng,
) -> Doc {
    let nb = 1usize << p.nbits();
    let pdim = dim / p.keys_per_byte();
    let stride = pdim + extra_stride;
    let mut packed = vec![0u8; ntok * stride];
    let mut buckets = Vec::with_capacity(ntok * dim);
    for t in 0..ntok {
        let row: Vec<usize> = (0..dim).map(|_| rng.below(nb)).collect();
        p.pack_row(&row, &mut packed[t * stride..t * stride + pdim]);
        // Garbage in the padding must never be read.
        for b in &mut packed[t * stride + pdim..(t + 1) * stride] {
            *b = 0xA5;
        }
        buckets.extend_from_slice(&row);
    }
    Doc {
        packed,
        buckets,
        codes: (0..ntok).map(|_| rng.below(ncent) as u32).collect(),
        inv: (0..ntok).map(|_| rng.f32(0.5, 1.5)).collect(),
        ntok,
        stride,
    }
}

/// f64 recomputation from the integer codes: what every kernel must approximate
/// to float rounding.
fn reference(
    lut: &Lut,
    q: &PreparedQuery,
    w: &[f32],
    doc: &Doc,
    cdot: Option<(&[f32], usize)>,
    use_inv: bool,
) -> f64 {
    let nq = q.n_tokens();
    let dim = q.dim();
    let mut total = 0.0f64;
    for qi in 0..nq {
        let mut best = f64::NEG_INFINITY;
        for t in 0..doc.ntok {
            let mut acc = 0i64;
            for d in 0..dim {
                let wq = (w[doc.buckets[t * dim + d]] / lut.scale())
                    .round()
                    .clamp(-127.0, 127.0) as i64;
                acc += q.codes()[qi * dim + d] as i64 * wq;
            }
            let c = match cdot {
                Some((cd, _)) => cd[doc.codes[t] as usize * nq + qi] as f64,
                None => 0.0,
            };
            let inv = if use_inv { doc.inv[t] as f64 } else { 1.0 };
            let s = (q.scales()[qi] as f64 * lut.scale() as f64 * acc as f64 + c) * inv;
            best = best.max(s);
        }
        total += best;
    }
    total
}

#[test]
fn simd_matches_scalar_bitwise_across_shapes() {
    let mut rng = Rng(0x243F6A8885A308D3);
    let mut simd_seen = false;
    // nq values exercise every fold branch: 1,3 pure tail; 7 one NEON block +
    // tail; 8 one AVX2 block; 9 block + 1-lane tail; 16 one AVX-512 block;
    // 32 the production shape.
    for &nq in &[1usize, 3, 7, 8, 9, 16, 32] {
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 16, 40, 48, 96, 128, 200, 256] {
                let p = ColbertPacking::new(nbits).unwrap();
                let w = weights(nbits, &mut rng);
                let lut = Lut::new(&p, &w).unwrap();
                let lut_scalar = lut.clone().force_scalar(true);
                let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
                let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
                let ncent = 12;
                let cdot: Vec<f32> = (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect();
                for &extra in &[0usize, 5] {
                    let doc = make_doc(&p, dim, 13, ncent, extra, &mut rng);
                    let view = DocView::new(&doc.packed, doc.ntok, doc.stride)
                        .codes(Codes::U32(&doc.codes))
                        .inv_norms(&doc.inv);
                    let fast = Scorer::new(&lut, &q)
                        .with_centroid_term(&cdot, ncent)
                        .unwrap()
                        .score(view);
                    let slow = Scorer::new(&lut_scalar, &q)
                        .with_centroid_term(&cdot, ncent)
                        .unwrap()
                        .score(view);
                    assert_eq!(
                        fast.to_bits(),
                        slow.to_bits(),
                        "nq={nq} nbits={nbits} dim={dim} extra={extra}: {} gave {fast}, scalar gave {slow}",
                        lut.kernel(dim)
                    );
                    simd_seen |= lut.kernel(dim).is_simd();
                    let want = reference(&lut, &q, &w, &doc, Some((&cdot, ncent)), true);
                    assert!(
                        (fast as f64 - want).abs() < 1e-3 * (1.0 + want.abs()),
                        "nq={nq} nbits={nbits} dim={dim}: {fast} vs f64 reference {want}"
                    );
                }
            }
        }
    }
    // On any machine with SIMD, the test must have exercised it; otherwise
    // the bit-equality above compared scalar with scalar.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let expect_simd = Lut::colbert(4, &[0.0; 16]).unwrap().kernel(128).is_simd();
        assert_eq!(
            simd_seen, expect_simd,
            "SIMD dispatch state did not match Lut::kernel"
        );
    }
    let _ = simd_seen;
}

#[test]
fn optional_terms_are_identities() {
    let mut rng = Rng(0x13198A2E03707344);
    for &nbits in &[1usize, 2, 4] {
        let dim = 128;
        let nq = 32;
        let p = ColbertPacking::new(nbits).unwrap();
        let w = weights(nbits, &mut rng);
        let lut = Lut::new(&p, &w).unwrap();
        let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
        let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
        let ncent = 9;
        let doc = make_doc(&p, dim, 21, ncent, 0, &mut rng);
        let zeros = vec![0.0f32; ncent * nq];
        let ones = vec![1.0f32; doc.ntok];

        let base = DocView::new(&doc.packed, doc.ntok, doc.stride);
        // No centroid term == zero centroid term (with any codes, or none).
        let a = Scorer::new(&lut, &q).score(base.inv_norms(&doc.inv));
        let b = Scorer::new(&lut, &q)
            .with_centroid_term(&zeros, ncent)
            .unwrap()
            .score(base.codes(Codes::U32(&doc.codes)).inv_norms(&doc.inv));
        assert_eq!(a.to_bits(), b.to_bits(), "nbits={nbits}: no-cdot vs zero-cdot");
        // No inv norms == inv norms of 1.
        let c = Scorer::new(&lut, &q).score(base);
        let d = Scorer::new(&lut, &q).score(base.inv_norms(&ones));
        assert_eq!(c.to_bits(), d.to_bits(), "nbits={nbits}: no-inv vs ones");
        // Code widths are interchangeable.
        let cdot: Vec<f32> = (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect();
        let s = Scorer::new(&lut, &q).with_centroid_term(&cdot, ncent).unwrap();
        let i64s: Vec<i64> = doc.codes.iter().map(|&c| c as i64).collect();
        let us: Vec<usize> = doc.codes.iter().map(|&c| c as usize).collect();
        let e = s.score(base.codes(Codes::U32(&doc.codes)));
        assert_eq!(e.to_bits(), s.score(base.codes(Codes::I64(&i64s))).to_bits());
        assert_eq!(e.to_bits(), s.score(base.codes(Codes::Usize(&us))).to_bits());
        let want = reference(&lut, &q, &w, &doc, Some((&cdot, ncent)), false);
        assert!((e as f64 - want).abs() < 1e-3 * (1.0 + want.abs()));
    }
}

#[test]
fn non_simd_dims_and_nbits8_score_on_scalar_path() {
    let mut rng = Rng(0xA4093822299F31D0);
    // nbits 2, dim 44: 11 packed bytes, not a multiple of 8 dims.
    for &(nbits, dim) in &[(2usize, 44usize), (4, 12), (8, 40), (8, 128)] {
        let p = ColbertPacking::new(nbits).unwrap();
        let w = weights(nbits, &mut rng);
        let lut = Lut::new(&p, &w).unwrap();
        let k = lut.kernel(dim);
        assert!(matches!(k, Kernel::Scalar(_)), "nbits={nbits} dim={dim}: {k}");
        let nq = 6;
        let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
        let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
        let ncent = 5;
        let cdot: Vec<f32> = (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect();
        let doc = make_doc(&p, dim, 9, ncent, 3, &mut rng);
        let got = Scorer::new(&lut, &q)
            .with_centroid_term(&cdot, ncent)
            .unwrap()
            .score(
                DocView::new(&doc.packed, doc.ntok, doc.stride)
                    .codes(Codes::U32(&doc.codes))
                    .inv_norms(&doc.inv),
            );
        let want = reference(&lut, &q, &w, &doc, Some((&cdot, ncent)), true);
        assert!(
            (got as f64 - want).abs() < 1e-3 * (1.0 + want.abs()),
            "nbits={nbits} dim={dim}: {got} vs {want}"
        );
    }
}

#[test]
fn shape_violations_are_errors_not_reads() {
    let lut = Lut::colbert(4, &(0..16).map(|i| i as f32 * 0.05 - 0.4).collect::<Vec<_>>()).unwrap();
    let dim = 64;
    let q = PreparedQuery::new(&lut, &vec![0.1; 8 * dim], 8, dim).unwrap();
    let cdot = vec![0.0f32; 4 * 8];
    let s = Scorer::new(&lut, &q).with_centroid_term(&cdot, 4).unwrap();
    let packed = vec![0u8; 3 * 32];
    let ok = DocView::new(&packed, 3, 32).codes(Codes::U32(&[0, 1, 3]));
    assert!(s.try_score(ok).is_ok());
    // Centroid id out of range.
    assert!(s.try_score(ok.codes(Codes::U32(&[0, 1, 4]))).is_err());
    assert!(s.try_score(ok.codes(Codes::I64(&[0, -1, 2]))).is_err());
    // Codes missing while a centroid term is set.
    assert!(s.try_score(ok.codes(Codes::None)).is_err());
    // Wrong code count, short buffer, narrow stride, wrong inv length.
    assert!(s.try_score(ok.codes(Codes::U32(&[0, 1]))).is_err());
    assert!(s
        .try_score(DocView::new(&packed[..80], 3, 32).codes(Codes::U32(&[0, 1, 3])))
        .is_err());
    assert!(s
        .try_score(DocView::new(&packed, 3, 31).codes(Codes::U32(&[0, 1, 3])))
        .is_err());
    assert!(s.try_score(ok.inv_norms(&[1.0, 1.0])).is_err());
    // Centroid term of the wrong size.
    assert!(Scorer::new(&lut, &q).with_centroid_term(&cdot, 5).is_err());
    // Query prepared against a different table.
    let other = Lut::colbert(4, &(0..16).map(|i| i as f32 * 0.07 - 0.5).collect::<Vec<_>>()).unwrap();
    assert!(Scorer::try_new(&other, &q).is_err());
    // Zero tokens scores zero and reads nothing.
    assert_eq!(s.score(DocView::new(&[], 0, 32).codes(Codes::U32(&[]))), 0.0);
}

#[test]
fn score_many_matches_score() {
    let mut rng = Rng(0x082EFA98EC4E6C89);
    let p = ColbertPacking::new(4).unwrap();
    let w = weights(4, &mut rng);
    let lut = Lut::new(&p, &w).unwrap();
    let (nq, dim) = (32, 128);
    let query: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
    let q = PreparedQuery::new(&lut, &query, nq, dim).unwrap();
    let docs: Vec<Doc> = (0..7)
        .map(|i| make_doc(&p, dim, 5 + 3 * i, 4, 0, &mut rng))
        .collect();
    let s = Scorer::new(&lut, &q);
    let mut out = vec![0.0f32; docs.len()];
    s.score_many(
        docs.iter().map(|d| DocView::new(&d.packed, d.ntok, d.stride)),
        &mut out,
    );
    for (d, &got) in docs.iter().zip(&out) {
        assert_eq!(
            got.to_bits(),
            s.score(DocView::new(&d.packed, d.ntok, d.stride)).to_bits()
        );
    }
}
