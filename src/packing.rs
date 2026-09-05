//! How a host packs `nbits`-wide bucket indices into bytes.
//!
//! This is the one thing a host has to tell the crate about its codec. The
//! fused table in [`crate::Lut`] is built by asking, for every byte value and
//! every key position, which bucket that position holds; nothing else about
//! the host's storage format leaks in.

use crate::Error;

/// Describes a bit-packing layout of `nbits`-wide bucket indices.
///
/// Key `k` of a byte is the `k`-th embedding dimension that byte carries, in
/// dimension order: a token's packed row `bytes[0..dim·nbits/8]` holds dims
/// `i·keys_per_byte + k` at `(bytes[i], key k)`.
pub trait Packing {
    /// Code width: 1, 2, 4 or 8.
    fn nbits(&self) -> usize;

    /// Bucket index (`0 .. 2^nbits`) stored at key position `key`
    /// (`0 .. 8/nbits`) of `byte`.
    fn bucket_index(&self, byte: u8, key: usize) -> usize;

    /// `8 / nbits`.
    fn keys_per_byte(&self) -> usize {
        8 / self.nbits()
    }

    /// Reference packer, the inverse of [`Packing::bucket_index`], for hosts
    /// that want to produce rows the same way the tests do. Not optimised.
    fn pack_row(&self, buckets: &[usize], out: &mut [u8]) {
        let kpb = self.keys_per_byte();
        assert_eq!(buckets.len() % kpb, 0, "buckets must fill whole bytes");
        assert!(out.len() >= buckets.len() / kpb, "output too short");
        for (i, chunk) in buckets.chunks(kpb).enumerate() {
            // Search the byte whose expansion matches; 256 candidates, and
            // this is a reference path, so brute force is fine.
            let byte = (0..=255u8)
                .find(|&b| (0..kpb).all(|k| self.bucket_index(b, k) == chunk[k]))
                .unwrap_or_else(|| {
                    panic!(
                        "no byte encodes buckets {chunk:?}: each must be < 2^nbits, and \
                         Packing::bucket_index must be a bijection over bytes"
                    )
                });
            out[i] = byte;
        }
    }
}

/// The ColBERT / PLAID residual layout, shared by ColBERTv2's `ResidualCodec`,
/// PLAID, fast-plaid, next-plaid and WARP.
///
/// `quantize_residuals` writes each dimension's bucket bits MSB-first into the
/// row, *bit 0 of the bucket first*. So within a byte, key `k` occupies bits
/// `7 - k·nbits` down to `8 - (k+1)·nbits`, and the bucket index is that
/// `nbits`-wide segment with its bits reversed. The original decoders express
/// this as a `byte_reversed_bits_map` followed by a group split; this is the
/// same function written directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColbertPacking {
    nbits: usize,
}

impl ColbertPacking {
    /// `nbits` must be 1, 2, 4 or 8.
    pub fn new(nbits: usize) -> Result<Self, Error> {
        match nbits {
            1 | 2 | 4 | 8 => Ok(Self { nbits }),
            n => Err(Error::NbitsUnsupported(n)),
        }
    }
}

impl Packing for ColbertPacking {
    fn nbits(&self) -> usize {
        self.nbits
    }

    #[inline]
    fn bucket_index(&self, byte: u8, key: usize) -> usize {
        let nbits = self.nbits;
        debug_assert!(key < 8 / nbits);
        let shift = 8 - nbits * (key + 1);
        let segment = (byte as usize >> shift) & ((1 << nbits) - 1);
        // Reverse the nbits-wide segment: the encoder emitted bucket bit 0
        // at the highest position of the group.
        let mut rev = 0usize;
        for b in 0..nbits {
            if segment & (1 << b) != 0 {
                rev |= 1 << (nbits - 1 - b);
            }
        }
        rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent re-implementation of next-plaid's `quantize_residuals`
    /// bit loop, so the packing here is pinned to the encoder, not to itself.
    fn encoder_pack(buckets: &[usize], nbits: usize) -> Vec<u8> {
        let mut out = vec![0u8; buckets.len() * nbits / 8];
        let mut bit_idx = 0usize;
        for &bucket in buckets {
            for b in 0..nbits {
                let bit = ((bucket >> b) & 1) as u8;
                out[bit_idx / 8] |= bit << (7 - (bit_idx % 8));
                bit_idx += 1;
            }
        }
        out
    }

    #[test]
    fn colbert_packing_matches_encoder_bit_loop() {
        for nbits in [1usize, 2, 4, 8] {
            let p = ColbertPacking::new(nbits).unwrap();
            let kpb = p.keys_per_byte();
            let n = 1usize << nbits;
            // Every bucket in every key position, plus a pseudo-random row.
            let mut buckets: Vec<usize> = Vec::new();
            for b in 0..n {
                for _ in 0..kpb {
                    buckets.push(b);
                }
            }
            let mut x = 0x9E3779B9u32;
            for _ in 0..(64 * kpb) {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                buckets.push(x as usize % n);
            }
            let bytes = encoder_pack(&buckets, nbits);
            for (d, &want) in buckets.iter().enumerate() {
                let got = p.bucket_index(bytes[d / kpb], d % kpb);
                assert_eq!(got, want, "nbits={nbits} dim={d}");
            }
            // And the reference packer round-trips.
            let mut repacked = vec![0u8; bytes.len()];
            p.pack_row(&buckets, &mut repacked);
            assert_eq!(repacked, bytes, "nbits={nbits}");
        }
    }
}
