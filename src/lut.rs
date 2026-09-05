//! The document-side table: packed byte → int8 bucket weights.

use crate::kernel::{self, Kernel};
use crate::packing::Packing;
use crate::Error;

/// Highest embedding dimension the kernels support. The SIMD expansion
/// buffer is `[i8; MAX_DIM]` and the AVX-512 dot reads it in 64-lane chunks,
/// which `dim ≤ 256` keeps in bounds for every byte-aligned `dim`.
pub const MAX_DIM: usize = 256;

/// The fused table factored per key position into 16-entry nibble tables,
/// the shape NEON `tbl` / SSE `pshufb` consume: one in-register lookup per
/// key position per 16 packed bytes.
///
/// Codes of width 1, 2 or 4 never cross a nibble boundary, so key `k` of a
/// byte is a function of exactly one of its nibbles. The factorisation is
/// *verified* over all 256 byte values when the [`Lut`] is built, so a
/// [`Packing`] the tables cannot represent falls back to the scalar path
/// instead of silently diverging from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NibbleTables {
    /// Per key position: weights indexed by the source nibble's value.
    pub tables: [[i8; 16]; 8],
    /// Whether key `k` reads the byte's high nibble (else the low one).
    pub from_hi: [bool; 8],
}

/// The document-side lookup state for one residual codec: a table turning
/// each packed residual byte directly into its `8/nbits` int8 bucket
/// weights, plus the dequantisation scale.
///
/// Build once per index (it depends only on the bucket weights and the
/// packing), share across threads.
#[derive(Debug, Clone)]
pub struct Lut {
    /// `[256 · keys_per_byte]` int8 weights; row `b` is the expansion of byte `b`.
    fused: Vec<i8>,
    keys_per_byte: usize,
    nbits: usize,
    /// `fused as f32 · scale ≈ bucket_weight`.
    scale: f32,
    nibble: Option<NibbleTables>,
    force_scalar: bool,
}

impl Lut {
    /// Build the table from a packing layout and the codec's `2^nbits` bucket
    /// weights (f32, in bucket-index order).
    ///
    /// Weights are quantised symmetrically to int8 with `scale = max|w| / 127`.
    pub fn new<P: Packing>(packing: &P, bucket_weights: &[f32]) -> Result<Self, Error> {
        let nbits = packing.nbits();
        if !matches!(nbits, 1 | 2 | 4 | 8) {
            return Err(Error::NbitsUnsupported(nbits));
        }
        let expected = 1usize << nbits;
        if bucket_weights.len() != expected {
            return Err(Error::BucketCount {
                expected,
                got: bucket_weights.len(),
            });
        }
        let max_abs = bucket_weights.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = (max_abs / 127.0).max(1e-12);
        let vals: Vec<i8> = bucket_weights
            .iter()
            .map(|&w| (w / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let keys_per_byte = 8 / nbits;
        let mut fused = vec![0i8; 256 * keys_per_byte];
        for byte in 0..256usize {
            for k in 0..keys_per_byte {
                let bi = packing.bucket_index(byte as u8, k);
                assert!(
                    bi < expected,
                    "Packing::bucket_index returned {bi} >= 2^nbits for byte {byte} key {k}"
                );
                fused[byte * keys_per_byte + k] = vals[bi];
            }
        }
        let nibble = derive_nibble_tables(&fused, keys_per_byte);
        Ok(Self {
            fused,
            keys_per_byte,
            nbits,
            scale,
            nibble,
            force_scalar: false,
        })
    }

    /// Convenience for the ColBERT / PLAID layout.
    pub fn colbert(nbits: usize, bucket_weights: &[f32]) -> Result<Self, Error> {
        Self::new(&crate::ColbertPacking::new(nbits)?, bucket_weights)
    }

    /// Pin every score to the scalar reference kernel. For tests and for
    /// measuring what the SIMD is worth; the results are bit-identical either
    /// way. The environment variable `MAXSIM_LUT_FORCE_SCALAR=1` has the same
    /// effect process-wide.
    pub fn force_scalar(mut self, yes: bool) -> Self {
        self.force_scalar = yes;
        self
    }

    /// Code width this table was built for.
    pub fn nbits(&self) -> usize {
        self.nbits
    }

    /// `8 / nbits`: how many dims one packed byte carries.
    pub fn keys_per_byte(&self) -> usize {
        self.keys_per_byte
    }

    /// Dequantisation scale: `fused as f32 · scale ≈ bucket_weight`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The `8/nbits` int8 weights a packed byte expands to, in dim order.
    #[inline]
    pub fn expand(&self, byte: u8) -> &[i8] {
        let base = byte as usize * self.keys_per_byte;
        &self.fused[base..base + self.keys_per_byte]
    }

    /// The whole fused table, `[256 · keys_per_byte]`, row `b` = byte `b`.
    pub fn fused_table(&self) -> &[i8] {
        &self.fused
    }

    /// The nibble-factored tables, if the layout admits them (always, for
    /// [`crate::ColbertPacking`] at nbits 1, 2 or 4; never at nbits 8).
    pub fn nibble_tables(&self) -> Option<&NibbleTables> {
        self.nibble.as_ref()
    }

    /// Which kernel [`crate::Scorer::score`] will run for this table and
    /// `dim` on this CPU. Print it next to any benchmark number: a speedup
    /// attributed to a path that never executed is the easiest measurement
    /// error to make and the hardest to notice.
    pub fn kernel(&self, dim: usize) -> Kernel {
        kernel::select(self, dim)
    }

    pub(crate) fn force_scalar_set(&self) -> bool {
        self.force_scalar
    }

    /// Two [`Lut`]s prepared from the same weights and packing are
    /// interchangeable for a [`crate::PreparedQuery`]; this is the identity
    /// the query checks.
    pub(crate) fn fingerprint(&self) -> (usize, u32) {
        (self.nbits, self.scale.to_bits())
    }
}

/// Factor `fused` into per-key nibble tables; `None` if any key position is
/// not a function of a single nibble.
fn derive_nibble_tables(fused: &[i8], keys_per_byte: usize) -> Option<NibbleTables> {
    if keys_per_byte > 8 || keys_per_byte == 1 {
        // nbits 8: a key spans the whole byte, no 16-entry factorisation.
        return None;
    }
    let mut tables = [[0i8; 16]; 8];
    let mut from_hi = [false; 8];
    for k in 0..keys_per_byte {
        let hi: [i8; 16] = std::array::from_fn(|x| fused[(x << 4) * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == hi[b >> 4]) {
            tables[k] = hi;
            from_hi[k] = true;
            continue;
        }
        let lo: [i8; 16] = std::array::from_fn(|x| fused[x * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == lo[b & 15]) {
            tables[k] = lo;
            from_hi[k] = false;
            continue;
        }
        return None;
    }
    Some(NibbleTables { tables, from_hi })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColbertPacking;

    fn weights(nbits: usize) -> Vec<f32> {
        let n = 1usize << nbits;
        (0..n)
            .map(|i| -0.35 + 0.7 * (i as f32 + 0.5) / n as f32)
            .collect()
    }

    #[test]
    fn fused_table_expands_to_quantised_bucket_weights() {
        for nbits in [1usize, 2, 4, 8] {
            let p = ColbertPacking::new(nbits).unwrap();
            let w = weights(nbits);
            let lut = Lut::new(&p, &w).unwrap();
            for byte in 0..=255u8 {
                for k in 0..lut.keys_per_byte() {
                    let bi = p.bucket_index(byte, k);
                    let want = (w[bi] / lut.scale()).round().clamp(-127.0, 127.0) as i8;
                    assert_eq!(lut.expand(byte)[k], want, "nbits={nbits} byte={byte} k={k}");
                }
            }
        }
    }

    #[test]
    fn nibble_factorisation_holds_for_sub_byte_codes() {
        for nbits in [1usize, 2, 4] {
            let lut = Lut::colbert(nbits, &weights(nbits)).unwrap();
            let nib = lut
                .nibble_tables()
                .unwrap_or_else(|| panic!("nbits={nbits}: not nibble-separable"));
            for b in 0..256usize {
                for k in 0..lut.keys_per_byte() {
                    let nibble = if nib.from_hi[k] { b >> 4 } else { b & 15 };
                    assert_eq!(
                        lut.fused_table()[b * lut.keys_per_byte() + k],
                        nib.tables[k][nibble]
                    );
                }
            }
        }
        assert!(Lut::colbert(8, &weights(8)).unwrap().nibble_tables().is_none());
    }

    #[test]
    fn rejects_bad_inputs() {
        assert_eq!(ColbertPacking::new(3).unwrap_err(), Error::NbitsUnsupported(3));
        assert_eq!(
            Lut::colbert(4, &[0.0; 15]).unwrap_err(),
            Error::BucketCount {
                expected: 16,
                got: 15
            }
        );
    }
}
