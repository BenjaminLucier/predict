//! Compressed sparse row (CSR) matrices: only the stored entries, row by row.

use std::borrow::Cow;

use ndarray::Array2;

use super::Absent;
use super::compressed::{Compressed, Names};
use crate::{Float, PredictError, Result};

const NAMES: Names = Names {
    offsets: "row_offsets",
    indices: "col_indices",
    lane: "row",
    index: "column",
};

/// A sparse matrix in compressed sparse row (CSR) form: the stored entries, row by
/// row.
///
/// Three arrays describe it. Row `i`'s entries are positions
/// `row_offsets[i]..row_offsets[i + 1]` of `col_indices` (their columns) and
/// `values`. For the 3 × 4 matrix
///
/// ```text
/// [ 5  0  0  1 ]
/// [ 0  0  2  0 ]
/// [ 0  3  0  0 ]
/// ```
///
/// they are `row_offsets = [0, 2, 3, 4]`, `col_indices = [0, 3, 2, 1]` and
/// `values = [5, 1, 2, 3]`. Entries that aren't stored are 0 unless
/// [`with_absent`](CsrMatrix::with_absent) says they are missing.
///
/// Like a dense array, each of the three is borrowed (`&[..]`, `&vec`) or moved in
/// (a `Vec`), never copied. The same matrix stored column by column is a
/// [`CscMatrix`](super::CscMatrix). See
/// <https://predict.rs/data.html#sparse-features>.
#[derive(Debug, Clone)]
pub struct CsrMatrix<'a, F: Float = f64> {
    rows: Compressed<'a, F>,
}

impl<'a, F: Float> CsrMatrix<'a, F> {
    /// A CSR matrix with `n_cols` columns, its structure checked. `row_offsets`
    /// has one entry per row plus one, so a matrix with no rows is `[0]`.
    /// `n_cols` is given because columns with no stored entries leave no trace in
    /// `col_indices`.
    ///
    /// The values are checked by the dataset's `validate_*` methods, like those
    /// of a dense array.
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// - `Malformed` with `context = "row_offsets"` if it is empty, doesn't start
    ///   at 0, or decreases.
    /// - `ShapeMismatch` with `context = "col_indices"` or `"values"` if either
    ///   doesn't have one entry per stored value, the last row offset.
    /// - `Malformed` with `context = "col_indices"` at the first row with a column
    ///   `>= n_cols`, or columns not strictly increasing (out of order or
    ///   repeated).
    pub fn new(
        n_cols: usize,
        row_offsets: impl Into<Cow<'a, [usize]>>,
        col_indices: impl Into<Cow<'a, [usize]>>,
        values: impl Into<Cow<'a, [F]>>,
    ) -> Result<Self> {
        let rows = Compressed::new(
            &NAMES,
            n_cols,
            row_offsets.into(),
            col_indices.into(),
            values.into(),
        )?;
        Ok(CsrMatrix { rows })
    }

    /// Set what entries that aren't stored mean: [`Absent::Zero`] (the default)
    /// or [`Absent::Missing`].
    pub fn with_absent(mut self, absent: Absent) -> Self {
        self.rows.absent = absent;
        self
    }

    pub fn n_rows(&self) -> usize {
        self.rows.n_lanes()
    }

    pub fn n_cols(&self) -> usize {
        self.rows.lane_len()
    }

    /// The number of stored entries.
    pub fn n_stored(&self) -> usize {
        self.rows.values().len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        self.rows.offsets()
    }

    pub fn col_indices(&self) -> &[usize] {
        self.rows.indices()
    }

    pub fn values(&self) -> &[F] {
        self.rows.values()
    }

    pub fn absent(&self) -> Absent {
        self.rows.absent
    }

    /// `NonFinite` at the first entry, in row order, that `is_rejected` picks out,
    /// its column counted from `col_offset`. With [`Absent::Missing`], entries that
    /// aren't stored are checked as NaN.
    pub(super) fn reject_first(
        &self,
        context: &'static str,
        col_offset: usize,
        is_rejected: impl Fn(F) -> bool,
    ) -> Result<()> {
        match self.rows.first_rejected(is_rejected) {
            Some((row, col, value)) => Err(PredictError::NonFinite {
                context,
                row,
                col: col_offset + col,
                value: value.into(),
            }),
            None => Ok(()),
        }
    }

    /// Every stored entry as `(row, col, value)`, row by row.
    pub(super) fn for_each_stored(&self, visit: impl FnMut(usize, usize, F)) {
        self.rows.for_each_stored(visit);
    }

    /// The same matrix with every entry written out: absent ones as 0 or NaN.
    pub(super) fn to_dense(&self) -> Array2<F> {
        self.rows.to_dense_by_lanes()
    }
}
