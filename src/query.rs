//! The query side: symmetric int8 codes, one scale per row, laid out the
//! way the kernels read them.

use crate::lut::Lut;
use crate::{padded_stride, Error, MAX_DIM};

/// A query quantised to int8 and pre-arranged for the kernels.
///
/// Built once per query against a [`Lut`]; scoring thousands of candidates
/// reuses it. `Send + Sync`, no interior mutability.
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    nq: usize,
    dim: usize,
    /// Row-major int8 codes in dim order, `[nq · dim]` (the scalar kernel's layout).
    values: Vec<i8>,
    /// Per-row `max|q| / 127`.
    scales: Vec<f32>,
    /// Codes permuted to *plane order* at a padded row stride: plane `k`
    /// holds the dims byte position `i` carries at key `k`
    /// (`d = i·keys_per_byte + k`), so the SIMD expand stores each `tbl`
    /// result contiguously. A dot product is permutation-invariant and the
    /// integer accumulator is order-invariant, so this changes no result.
    /// Only the SIMD kernels read these; on targets without one they are
    /// still built (cheap, `nq · 64..256` bytes) so the layout stays tested.
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
    planes: Vec<i8>,
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
    stride: usize,
    /// Per row: `scales[q] · lut.scale`, the query-constant factor the fold
    /// applies to each integer accumulator.
    sqw: Vec<f32>,
    /// `[nq]` zeros: the centroid row used when the host supplies no centroid term.
    zeros: Vec<f32>,
    lut_fingerprint: (usize, u32),
}

impl PreparedQuery {
    /// Quantise a query of `n_tokens` rows × `dim` (row-major f32).
    ///
    /// Each row is scaled by `max|q| / 127` so its largest component maps to
    /// ±127; an all-zero row gets scale 0 and codes 0.
    pub fn new(lut: &Lut, query: &[f32], n_tokens: usize, dim: usize) -> Result<Self, Error> {
        if dim > MAX_DIM {
            return Err(Error::DimTooLarge(dim));
        }
        if !(dim * lut.nbits()).is_multiple_of(8) {
            return Err(Error::DimNotByteAligned {
                dim,
                nbits: lut.nbits(),
            });
        }
        if query.len() != n_tokens * dim {
            return Err(Error::Shape(format!(
                "query has {} values, expected n_tokens {n_tokens} × dim {dim}",
                query.len()
            )));
        }
        let nq = n_tokens;
        let mut values = vec![0i8; nq * dim];
        let mut scales = vec![0.0f32; nq];
        for (qi, row) in query.chunks_exact(dim).enumerate() {
            let max_abs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            if max_abs <= 0.0 {
                continue;
            }
            let scale = max_abs / 127.0;
            scales[qi] = scale;
            for (d, &x) in row.iter().enumerate() {
                values[qi * dim + d] = (x / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let kpb = lut.keys_per_byte();
        let pdim = dim / kpb;
        let stride = padded_stride(dim);
        let mut planes = vec![0i8; nq * stride];
        for qi in 0..nq {
            let row = &values[qi * dim..(qi + 1) * dim];
            let out = &mut planes[qi * stride..qi * stride + dim];
            for i in 0..pdim {
                for k in 0..kpb {
                    out[k * pdim + i] = row[i * kpb + k];
                }
            }
        }
        let sqw = scales.iter().map(|&s| s * lut.scale()).collect();
        Ok(Self {
            nq,
            dim,
            values,
            scales,
            planes,
            stride,
            sqw,
            zeros: vec![0.0f32; nq],
            lut_fingerprint: lut.fingerprint(),
        })
    }

    /// Number of query tokens (rows).
    pub fn n_tokens(&self) -> usize {
        self.nq
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row-major int8 codes, `[n_tokens · dim]`.
    pub fn codes(&self) -> &[i8] {
        &self.values
    }

    /// Per-row dequantisation scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn planes(&self) -> &[i8] {
        &self.planes
    }
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }
    pub(crate) fn sqw(&self) -> &[f32] {
        &self.sqw
    }
    pub(crate) fn zeros(&self) -> &[f32] {
        &self.zeros
    }
    pub(crate) fn matches(&self, lut: &Lut) -> bool {
        self.lut_fingerprint == lut.fingerprint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantisation_is_symmetric_per_row() {
        let lut = Lut::colbert(4, &[0.0; 16]).unwrap();
        let dim = 8;
        let q = vec![
            0.5, -1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let p = PreparedQuery::new(&lut, &q, 2, dim).unwrap();
        assert_eq!(p.codes()[..3], [64, -127, 32]);
        assert!((p.scales()[0] - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(p.scales()[1], 0.0);
        assert!(p.codes()[dim..].iter().all(|&c| c == 0));
        // Planes: nbits 4 → 2 keys/byte, plane 0 = even dims, plane 1 = odd dims.
        assert_eq!(p.planes()[..4], [64, 32, 0, 0]);
        assert_eq!(p.planes()[4..8], [-127, 0, 0, 0]);
        assert_eq!(p.stride(), 64);
    }

    #[test]
    fn rejects_misaligned_dim() {
        let lut = Lut::colbert(2, &[0.0; 4]).unwrap();
        assert_eq!(
            PreparedQuery::new(&lut, &[0.0; 10], 1, 10).unwrap_err(),
            Error::DimNotByteAligned { dim: 10, nbits: 2 }
        );
        assert_eq!(
            PreparedQuery::new(&lut, &[0.0; 264], 1, 264).unwrap_err(),
            Error::DimTooLarge(264)
        );
    }
}
