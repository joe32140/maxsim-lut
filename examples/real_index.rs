//! Score a real next-plaid index with the crate: parity against float
//! decompression, ns/token on real residual codes, and how loose the exact
//! per-token skip bound is on real data.
//!
//!   cargo run --release --example real_index -- <index_dir> <queries.npy> [n_queries] [ref_docs]
//!
//! `index_dir` is a next-plaid ≥1.7 index (`metadata.json`, `centroids.npy`,
//! `bucket_weights.npy`, `<i>.codes.npy`, `<i>.residuals.npy`,
//! `<i>.inv_norms.npy`, `doclens.<i>.json`). `queries.npy` is f32
//! `[n, n_tokens, dim]`. No dependencies: a 60-line NPY reader below.

// The float reference deliberately spells out the index arithmetic.
#![allow(clippy::needless_range_loop)]

use std::fs;
use std::path::Path;
use std::time::Instant;

use maxsim_lut::{Codes, DocView, Lut, PreparedQuery, Scorer};

/// Minimal NPY (v1/v2, C order) reader: returns (shape, dtype descr, raw bytes).
fn read_npy(path: &Path) -> (Vec<usize>, String, Vec<u8>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}: not an npy file", path.display());
    let (hlen, hstart) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).unwrap();
    let descr = header
        .split("'descr':")
        .nth(1)
        .unwrap()
        .split('\'')
        .nth(1)
        .unwrap()
        .to_string();
    assert!(
        header.contains("'fortran_order': False"),
        "{}: Fortran order unsupported",
        path.display()
    );
    let shape_str = header
        .split("'shape':")
        .nth(1)
        .unwrap()
        .split('(')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    (shape, descr, bytes[hstart + hlen..].to_vec())
}

fn npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let (shape, descr, raw) = read_npy(path);
    assert_eq!(descr, "<f4", "{}", path.display());
    (
        shape,
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn npy_u8(path: &Path) -> (Vec<usize>, Vec<u8>) {
    let (shape, descr, raw) = read_npy(path);
    assert_eq!(descr, "|u1", "{}", path.display());
    (shape, raw)
}

fn npy_codes(path: &Path) -> Vec<u32> {
    let (_, descr, raw) = read_npy(path);
    match descr.as_str() {
        "<i8" => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect(),
        "<u4" | "<i4" => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        d => panic!("{}: unsupported code dtype {d}", path.display()),
    }
}

/// `"key": <int>` out of a small JSON file, no parser.
fn json_usize(text: &str, key: &str) -> usize {
    let pat = format!("\"{key}\":");
    let rest = &text[text.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len()..];
    rest.trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: real_index <index_dir> <queries.npy> [n_queries=20] [ref_docs=200]");
        std::process::exit(2);
    }
    let dir = Path::new(&args[0]);
    let n_queries: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let ref_docs: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);

    let meta = fs::read_to_string(dir.join("metadata.json")).unwrap();
    let nbits = json_usize(&meta, "nbits");
    let num_chunks = json_usize(&meta, "num_chunks");
    let dim = json_usize(&meta, "embedding_dim");
    let (cshape, centroids) = npy_f32(&dir.join("centroids.npy"));
    let ncent = cshape[0];
    assert_eq!(cshape[1], dim);
    let (_, weights) = npy_f32(&dir.join("bucket_weights.npy"));
    assert_eq!(weights.len(), 1 << nbits);

    let mut codes: Vec<u32> = Vec::new();
    let mut inv: Vec<f32> = Vec::new();
    let mut packed: Vec<u8> = Vec::new();
    let mut doclens: Vec<usize> = Vec::new();
    let pdim = dim * nbits / 8;
    for c in 0..num_chunks {
        codes.extend(npy_codes(&dir.join(format!("{c}.codes.npy"))));
        inv.extend(npy_f32(&dir.join(format!("{c}.inv_norms.npy"))).1);
        let (rshape, r) = npy_u8(&dir.join(format!("{c}.residuals.npy")));
        assert_eq!(rshape[1], pdim, "residual row width");
        packed.extend(r);
        let dl = fs::read_to_string(dir.join(format!("doclens.{c}.json"))).unwrap();
        doclens.extend(
            dl.trim()
                .trim_matches(['[', ']'])
                .split(',')
                .map(|s| s.trim().parse::<usize>().unwrap()),
        );
    }
    let ntok_total: usize = doclens.iter().sum();
    assert_eq!(codes.len(), ntok_total);
    assert_eq!(inv.len(), ntok_total);
    assert_eq!(packed.len(), ntok_total * pdim);
    let mut offsets = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0usize);
    for &l in &doclens {
        offsets.push(offsets.last().unwrap() + l);
    }

    let (qshape, queries) = npy_f32(Path::new(&args[1]));
    let (nq_rows, qdim) = (qshape[1], qshape[2]);
    assert_eq!(qdim, dim);
    let n_queries = n_queries.min(qshape[0]);

    let lut = Lut::colbert(nbits, &weights).unwrap();
    println!(
        "index: {} docs, {} tokens, dim {dim}, nbits {nbits}, {ncent} centroids, {} B/token packed\nkernel: {}\nqueries: {n_queries} × {nq_rows} rows",
        doclens.len(),
        ntok_total,
        pdim,
        lut.kernel(dim)
    );
    let max_w = weights.iter().fold(0f32, |m, &w| m.max(w.abs()));

    let mut t_cdot = 0.0f64;
    let mut t_score = 0.0f64;
    let mut max_abs_err = 0f32;
    let mut sum_rel_err = 0f64;
    let mut n_ref = 0usize;
    let mut skippable = 0usize;
    let mut skip_seen = 0usize;
    let mut bound_sum = 0f64;
    let mut resid_abs_sum = 0f64;
    let mut resid_n = 0usize;

    for qi in 0..n_queries {
        let qf = &queries[qi * nq_rows * dim..(qi + 1) * nq_rows * dim];
        let q = PreparedQuery::new(&lut, qf, nq_rows, dim).unwrap();

        // Stage-1 product the host would already have: centroid-major [ncent × nq].
        let t = Instant::now();
        let mut cdot = vec![0f32; ncent * nq_rows];
        for c in 0..ncent {
            let cv = &centroids[c * dim..(c + 1) * dim];
            for r in 0..nq_rows {
                let qr = &qf[r * dim..(r + 1) * dim];
                cdot[c * nq_rows + r] = cv.iter().zip(qr).map(|(a, b)| a * b).sum();
            }
        }
        t_cdot += t.elapsed().as_secs_f64();

        let scorer = Scorer::new(&lut, &q).with_centroid_term(&cdot, ncent).unwrap();
        let mut scores = vec![0f32; doclens.len()];
        let t = Instant::now();
        for (d, s) in scores.iter_mut().enumerate() {
            let (a, b) = (offsets[d], offsets[d + 1]);
            *s = scorer.score(
                DocView::new(&packed[a * pdim..b * pdim], b - a, pdim)
                    .codes(Codes::U32(&codes[a..b]))
                    .inv_norms(&inv[a..b]),
            );
        }
        t_score += t.elapsed().as_secs_f64();

        // Float reference on the first `ref_docs` docs: decompress token =
        // centroid + bucket weight per dim, exact f32 MaxSim with inv norms.
        let mut tok = vec![0f32; dim];
        for d in 0..ref_docs.min(doclens.len()) {
            let (a, b) = (offsets[d], offsets[d + 1]);
            let mut best = vec![f32::NEG_INFINITY; nq_rows];
            for t in a..b {
                let cv = &centroids[codes[t] as usize * dim..(codes[t] as usize + 1) * dim];
                let row = &packed[t * pdim..(t + 1) * pdim];
                for dd in 0..dim {
                    // ColBERT packing: key k of byte i is bits (7 - k*nbits ..), bucket bit-reversed.
                    let kpb = 8 / nbits;
                    let (i, k) = (dd / kpb, dd % kpb);
                    let shift = 8 - nbits * (k + 1);
                    let seg = (row[i] >> shift) as usize & ((1 << nbits) - 1);
                    let mut bucket = 0usize;
                    for bit in 0..nbits {
                        if seg & (1 << bit) != 0 {
                            bucket |= 1 << (nbits - 1 - bit);
                        }
                    }
                    tok[dd] = cv[dd] + weights[bucket];
                }
                for r in 0..nq_rows {
                    let qr = &qf[r * dim..(r + 1) * dim];
                    let s: f32 = tok.iter().zip(qr).map(|(x, y)| x * y).sum::<f32>() * inv[t];
                    if s > best[r] {
                        best[r] = s;
                    }
                }
            }
            let reference: f32 = best.iter().sum();
            let err = (scores[d] - reference).abs();
            max_abs_err = max_abs_err.max(err);
            sum_rel_err += (err / reference.abs().max(1e-6)) as f64;
            n_ref += 1;
        }

        // Exact skip bound: |residual term| <= sqw[r] * 127 * Σ|q̂| = max|w| * ||q_r||_1
        // (up to quantisation). Simulate a sequential pass over each doc's tokens and
        // count tokens where every row could have been skipped.
        let l1: Vec<f32> = (0..nq_rows)
            .map(|r| qf[r * dim..(r + 1) * dim].iter().map(|x| x.abs()).sum())
            .collect();
        let bounds: Vec<f32> = l1.iter().map(|&l| l * max_w).collect();
        bound_sum += bounds.iter().map(|&b| b as f64).sum::<f64>() / nq_rows as f64;
        for d in 0..ref_docs.min(doclens.len()) {
            let (a, b) = (offsets[d], offsets[d + 1]);
            let mut best = vec![f32::NEG_INFINITY; nq_rows];
            for t in a..b {
                let crow = &cdot[codes[t] as usize * nq_rows..(codes[t] as usize + 1) * nq_rows];
                let can_skip = (0..nq_rows).all(|r| (crow[r] + bounds[r]) * inv[t] <= best[r]);
                skip_seen += 1;
                if can_skip {
                    skippable += 1;
                }
                // Actual residual term magnitude for the record.
                let cv = &centroids[codes[t] as usize * dim..(codes[t] as usize + 1) * dim];
                let row = &packed[t * pdim..(t + 1) * pdim];
                let kpb = 8 / nbits;
                for r in 0..nq_rows {
                    let qr = &qf[r * dim..(r + 1) * dim];
                    let mut resid = 0f32;
                    for dd in 0..dim {
                        let (i, k) = (dd / kpb, dd % kpb);
                        let shift = 8 - nbits * (k + 1);
                        let seg = (row[i] >> shift) as usize & ((1 << nbits) - 1);
                        let mut bucket = 0usize;
                        for bit in 0..nbits {
                            if seg & (1 << bit) != 0 {
                                bucket |= 1 << (nbits - 1 - bit);
                            }
                        }
                        resid += qr[dd] * weights[bucket];
                    }
                    resid_abs_sum += resid.abs() as f64;
                    resid_n += 1;
                    let s =
                        (crow[r] + (cv.iter().zip(qr).map(|(x, y)| x * y).sum::<f32>() - crow[r]) + resid)
                            * inv[t];
                    if s > best[r] {
                        best[r] = s;
                    }
                }
            }
        }
    }

    let tokens_scored = ntok_total as f64 * n_queries as f64;
    println!(
        "\nexhaustive stage-2 over all docs: {:.2} ms/query, {:.2} ns/token (kernel only; stage-1 cdot GEMM excluded: {:.1} ms/query naive)",
        t_score / n_queries as f64 * 1e3,
        t_score / tokens_scored * 1e9,
        t_cdot / n_queries as f64 * 1e3
    );
    println!(
        "parity vs float decompression on {n_ref} (query, doc) pairs: max |Δscore| {max_abs_err:.4}, mean rel {:.2e}",
        sum_rel_err / n_ref.max(1) as f64
    );
    println!(
        "exact skip bound: mean bound {:.3} vs mean |residual term| {:.4}; tokens skippable {}/{} ({:.2}%)",
        bound_sum / n_queries as f64,
        resid_abs_sum / resid_n.max(1) as f64,
        skippable,
        skip_seen,
        100.0 * skippable as f64 / skip_seen.max(1) as f64
    );
}
