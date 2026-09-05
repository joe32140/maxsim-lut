//! Scoring from many threads at once.
//!
//! The crate owns no thread pool and tells hosts to fan out across candidate
//! chunks themselves, which makes three things load-bearing and none of them
//! visible to a single-threaded test:
//!
//! * [`Lut`], [`PreparedQuery`] and [`Scorer`] must really be `Send + Sync`;
//! * the per-thread kernel scratch must not leak state between threads;
//! * first-use calibration must be decided once, under a race, and every
//!   thread must see the same answer.
//!
//! This file is its own integration-test binary, so it gets a fresh process
//! and its threads genuinely contend for that first dispatch rather than
//! finding it already resolved by an earlier test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use maxsim_lut::{Codes, ColbertPacking, DocView, Kernel, Lut, Packing, PreparedQuery, Scorer};

const THREADS: usize = 8;

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

struct Fixture {
    lut: Lut,
    query: PreparedQuery,
    cdot: Vec<f32>,
    packed: Vec<u8>,
    codes: Vec<u32>,
    inv: Vec<f32>,
    ndocs: usize,
    ntok: usize,
    pdim: usize,
    ncent: usize,
}

impl Fixture {
    fn new() -> Self {
        let (nbits, dim, nq, ntok, ndocs, ncent) = (4usize, 128usize, 32usize, 24usize, 64usize, 11usize);
        let mut rng = Rng(0x6C8E9CF570932BD5);
        let p = ColbertPacking::new(nbits).unwrap();
        let mut w: Vec<f32> = (0..1 << nbits).map(|_| rng.f32(-0.4, 0.4)).collect();
        w.sort_by(|a, b| a.total_cmp(b));
        let lut = Lut::new(&p, &w).unwrap();
        let q: Vec<f32> = (0..nq * dim).map(|_| rng.f32(-1.0, 1.0)).collect();
        let query = PreparedQuery::new(&lut, &q, nq, dim).unwrap();
        let pdim = dim / p.keys_per_byte();
        Self {
            cdot: (0..ncent * nq).map(|_| rng.f32(-1.0, 1.0)).collect(),
            packed: (0..ndocs * ntok * pdim)
                .map(|_| (rng.next() >> 56) as u8)
                .collect(),
            codes: (0..ndocs * ntok)
                .map(|_| (rng.next() % ncent as u64) as u32)
                .collect(),
            inv: (0..ndocs * ntok).map(|_| rng.f32(0.8, 1.2)).collect(),
            lut,
            query,
            ndocs,
            ntok,
            pdim,
            ncent,
        }
    }

    fn doc(&self, d: usize) -> DocView<'_> {
        let (a, b) = (d * self.ntok, (d + 1) * self.ntok);
        DocView::new(&self.packed[a * self.pdim..b * self.pdim], self.ntok, self.pdim)
            .codes(Codes::U32(&self.codes[a..b]))
            .inv_norms(&self.inv[a..b])
    }

    fn scorer(&self) -> Scorer<'_> {
        Scorer::new(&self.lut, &self.query)
            .with_centroid_term(&self.cdot, self.ncent)
            .unwrap()
    }

    fn score_all(&self) -> Vec<f32> {
        let s = self.scorer();
        (0..self.ndocs).map(|d| s.score(self.doc(d))).collect()
    }
}

/// Every thread scores every document and must agree, bit for bit, with a
/// single-threaded run. A shared scratch buffer or a half-initialised
/// dispatch would show up here as a mismatch on some thread.
#[test]
fn concurrent_scoring_matches_single_threaded_bitwise() {
    let f = Fixture::new();
    // Threads start together so their first scores contend for calibration.
    let gate = Arc::new(Barrier::new(THREADS));
    let results: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let f = &f;
                s.spawn(move || {
                    gate.wait();
                    // Several passes so threads keep overlapping, not just
                    // race once at startup.
                    let mut last = f.score_all();
                    for _ in 0..4 {
                        let again = f.score_all();
                        assert_eq!(
                            last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "one thread's own repeated runs disagreed"
                        );
                        last = again;
                    }
                    last
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    // The reference is computed after the threads, on the calibrated kernel,
    // and again forced through the scalar path.
    let want = f.score_all();
    let scalar_lut = f.lut.clone().force_scalar(true);
    let scalar_scorer = Scorer::new(&scalar_lut, &f.query)
        .with_centroid_term(&f.cdot, f.ncent)
        .unwrap();
    for (d, w) in want.iter().enumerate() {
        assert_eq!(
            w.to_bits(),
            scalar_scorer.score(f.doc(d)).to_bits(),
            "doc {d}: threaded-era kernel disagrees with the scalar reference"
        );
    }
    for (t, got) in results.iter().enumerate() {
        for (d, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "thread {t}, doc {d}: {g} vs {w}");
        }
    }
}

/// Calibration is a one-shot decision behind a `OnceLock`. Under a thundering
/// herd every thread must observe the same kernel, and it must be one this
/// CPU can actually run.
#[test]
fn concurrent_dispatch_resolves_to_one_kernel() {
    let f = Fixture::new();
    let gate = Arc::new(Barrier::new(THREADS));
    let calls = Arc::new(AtomicUsize::new(0));
    let seen: Vec<Kernel> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let (gate, calls) = (Arc::clone(&gate), Arc::clone(&calls));
                let f = &f;
                s.spawn(move || {
                    gate.wait();
                    let mut k = f.lut.kernel(f.query.dim());
                    for _ in 0..64 {
                        let again = f.lut.kernel(f.query.dim());
                        assert_eq!(k, again, "dispatch changed its mind within a thread");
                        k = again;
                        calls.fetch_add(1, Ordering::Relaxed);
                    }
                    k
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    assert_eq!(calls.load(Ordering::Relaxed), THREADS * 64);
    let first = seen[0];
    assert!(
        seen.iter().all(|&k| k == first),
        "threads disagreed about the kernel: {seen:?}"
    );
    assert!(
        maxsim_lut::supported_kernels().contains(&first) || !first.is_simd(),
        "{first} is not executable on this CPU"
    );
}

/// A `Scorer` is `Copy` and shared by reference above; confirm the owned
/// pieces really carry the marker traits the docs promise, so a host can put
/// them in an `Arc` or hand them to a work-stealing pool.
#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Lut>();
    assert_send_sync::<PreparedQuery>();
    assert_send_sync::<Scorer<'static>>();
    assert_send_sync::<DocView<'static>>();
    assert_send_sync::<Codes<'static>>();
    assert_send_sync::<Kernel>();
}
