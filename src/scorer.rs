//! Binding a table and a query to score documents.

use crate::kernel::{self, Args};
use crate::lut::Lut;
use crate::query::PreparedQuery;
use crate::Error;

/// A document's per-token centroid ids, in whatever integer width the host
/// stores them. [`Codes::None`] is valid when the [`Scorer`] carries no
/// centroid term.
#[derive(Debug, Clone, Copy)]
pub enum Codes<'a> {
    /// No centroid ids (only valid without a centroid term).
    None,
    /// `u32` ids.
    U32(&'a [u32]),
    /// `i64` ids (next-plaid's on-disk width). Negative values fail range checks.
    I64(&'a [i64]),
    /// `usize` ids.
    Usize(&'a [usize]),
}

impl Codes<'_> {
    #[inline(always)]
    pub(crate) fn id(&self, t: usize) -> usize {
        match self {
            Codes::None => 0,
            Codes::U32(s) => s[t] as usize,
            // A negative i64 wraps to a huge usize and fails the range check.
            Codes::I64(s) => s[t] as usize,
            Codes::Usize(s) => s[t],
        }
    }

    fn len(&self) -> Option<usize> {
        match self {
            Codes::None => None,
            Codes::U32(s) => Some(s.len()),
            Codes::I64(s) => Some(s.len()),
            Codes::Usize(s) => Some(s.len()),
        }
    }
}

/// One candidate document, borrowed from wherever the host keeps it.
#[derive(Debug, Clone, Copy)]
pub struct DocView<'a> {
    /// Packed residual rows, one per token, contiguous at `row_stride` bytes.
    /// Must hold at least `n_tokens · row_stride` bytes.
    pub packed: &'a [u8],
    /// Number of tokens.
    pub n_tokens: usize,
    /// Bytes from one token's row to the next; at least `dim / keys_per_byte`.
    pub row_stride: usize,
    /// Per-token centroid ids, indexing the scorer's centroid term.
    pub codes: Codes<'a>,
    /// Optional per-token `1 / ‖reconstructed token‖`. `None` scores the
    /// unnormalised reconstruction (multiplies by 1).
    pub inv_norms: Option<&'a [f32]>,
}

impl<'a> DocView<'a> {
    /// A document with no centroid ids and no normalisation; add them with
    /// [`DocView::codes`] and [`DocView::inv_norms`].
    pub fn new(packed: &'a [u8], n_tokens: usize, row_stride: usize) -> Self {
        Self {
            packed,
            n_tokens,
            row_stride,
            codes: Codes::None,
            inv_norms: None,
        }
    }

    /// Attach per-token centroid ids.
    pub fn codes(mut self, codes: Codes<'a>) -> Self {
        self.codes = codes;
        self
    }

    /// Attach per-token inverse norms.
    pub fn inv_norms(mut self, inv: &'a [f32]) -> Self {
        self.inv_norms = Some(inv);
        self
    }
}

/// A [`Lut`] and a [`PreparedQuery`] bound together, with the host's
/// optional centroid term, ready to score documents.
///
/// `Copy`, `Send + Sync`: make one per query and share it across the
/// threads scoring that query's candidates.
#[derive(Debug, Clone, Copy)]
pub struct Scorer<'a> {
    lut: &'a Lut,
    query: &'a PreparedQuery,
    cdot: Option<&'a [f32]>,
    num_centroids: usize,
}

impl<'a> Scorer<'a> {
    /// Bind a table and a query. Panics if the query was prepared against a
    /// different table (use [`Scorer::try_new`] to get an error instead).
    pub fn new(lut: &'a Lut, query: &'a PreparedQuery) -> Self {
        Self::try_new(lut, query).expect("PreparedQuery was built against a different Lut")
    }

    /// Bind a table and a query.
    pub fn try_new(lut: &'a Lut, query: &'a PreparedQuery) -> Result<Self, Error> {
        if !query.matches(lut) {
            return Err(Error::LutMismatch);
        }
        Ok(Self {
            lut,
            query,
            cdot: None,
            num_centroids: 0,
        })
    }

    /// Supply the host's query × centroid scores, **centroid-major**:
    /// `cdot[cid · n_query_tokens + q]`. One centroid's scores across all
    /// query rows are then contiguous, so the vectorised fold loads them as
    /// one vector. Hosts that hold the `[n_query_tokens, num_centroids]`
    /// orientation transpose once per query.
    pub fn with_centroid_term(
        mut self,
        cdot_centroid_major: &'a [f32],
        num_centroids: usize,
    ) -> Result<Self, Error> {
        let nq = self.query.n_tokens();
        if cdot_centroid_major.len() != num_centroids * nq {
            return Err(Error::Shape(format!(
                "centroid term has {} values, expected num_centroids {num_centroids} × n_query_tokens {nq}",
                cdot_centroid_major.len()
            )));
        }
        self.cdot = Some(cdot_centroid_major);
        self.num_centroids = num_centroids;
        Ok(self)
    }

    /// The table this scorer uses.
    pub fn lut(&self) -> &'a Lut {
        self.lut
    }

    /// The query this scorer uses.
    pub fn query(&self) -> &'a PreparedQuery {
        self.query
    }

    /// MaxSim of the query against one document. Panics on a shape
    /// violation (see [`Scorer::try_score`] for the checked form); every
    /// SIMD path returns the same bits as the scalar reference.
    #[inline]
    pub fn score(&self, doc: DocView<'_>) -> f32 {
        match self.try_score(doc) {
            Ok(s) => s,
            Err(e) => panic!("maxsim_lut::Scorer::score: {e}"),
        }
    }

    /// MaxSim of the query against one document, with shape validation.
    pub fn try_score(&self, doc: DocView<'_>) -> Result<f32, Error> {
        let args = self.validate(doc)?;
        Ok(kernel::maxsim(self.lut, &args))
    }

    /// Score many documents into `out` (`out.len()` must equal the number of
    /// documents). Sequential; wrap the call in the host's parallel iterator
    /// over chunks to fan out.
    ///
    /// Panics if the counts disagree in either direction. Silently dropping
    /// the tail of a candidate list is the kind of bug that shows up as a
    /// slightly worse recall number months later, not as a failure.
    pub fn score_many<'d, I>(&self, docs: I, out: &mut [f32])
    where
        I: IntoIterator<Item = DocView<'d>>,
    {
        let slots = out.len();
        let mut n = 0usize;
        for doc in docs {
            let slot = out
                .get_mut(n)
                .unwrap_or_else(|| panic!("score_many: more than {slots} documents"));
            *slot = self.score(doc);
            n += 1;
        }
        assert_eq!(n, slots, "score_many: {n} documents for {slots} output slots");
    }

    fn validate<'d>(&self, doc: DocView<'d>) -> Result<Args<'d>, Error>
    where
        'a: 'd,
    {
        let q = self.query;
        let dim = q.dim();
        let kpb = self.lut.keys_per_byte();
        let pdim = dim / kpb;
        if doc.row_stride < pdim {
            return Err(Error::Shape(format!(
                "row_stride {} < {pdim} packed bytes for dim {dim} at {kpb} keys/byte",
                doc.row_stride
            )));
        }
        if doc.n_tokens > 0 && doc.packed.len() < (doc.n_tokens - 1) * doc.row_stride + pdim {
            return Err(Error::Shape(format!(
                "packed has {} bytes, need {} for {} tokens at stride {}",
                doc.packed.len(),
                (doc.n_tokens - 1) * doc.row_stride + pdim,
                doc.n_tokens,
                doc.row_stride
            )));
        }
        if let Some(inv) = doc.inv_norms {
            if inv.len() != doc.n_tokens {
                return Err(Error::Shape(format!(
                    "inv_norms has {} values for {} tokens",
                    inv.len(),
                    doc.n_tokens
                )));
            }
        }
        let (cdot, cdot_stride) = match self.cdot {
            Some(c) => {
                let n = doc.codes.len().ok_or_else(|| {
                    Error::Shape("centroid term supplied but DocView has Codes::None".into())
                })?;
                if n != doc.n_tokens {
                    return Err(Error::Shape(format!(
                        "codes has {n} ids for {} tokens",
                        doc.n_tokens
                    )));
                }
                for t in 0..doc.n_tokens {
                    let cid = doc.codes.id(t);
                    if cid >= self.num_centroids {
                        return Err(Error::Shape(format!(
                            "centroid id {cid} at token {t} out of range {}",
                            self.num_centroids
                        )));
                    }
                }
                (c, q.n_tokens())
            }
            // No centroid term: every token reads the same zero row.
            None => (q.zeros(), 0usize),
        };
        Ok(Args {
            query: q,
            packed: doc.packed,
            row_stride: doc.row_stride,
            n_tokens: doc.n_tokens,
            codes: if self.cdot.is_some() {
                doc.codes
            } else {
                Codes::None
            },
            cdot,
            cdot_stride,
            inv_norms: doc.inv_norms,
        })
    }
}
