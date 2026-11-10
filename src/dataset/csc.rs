//! Compressed sparse column (CSC) matrices: only the stored entries, column by
//! column.

use std::borrow::Cow;

use ndarray::Array2;

use super::Absent;
use super::compressed::{Compressed, Names};
use crate::{Float, PredictError, Result};

const NAMES: Names = Names {
    offsets: "col_offsets",
    indices: "row_indices",
    lane: "column",
    index: "row",
};

/// A sparse matrix in compressed sparse column (CSC) form: the stored entries,
/// column by column. The same matrix stored row by row is a
/// [`CsrMatrix`](super::CsrMatrix), with the roles of rows and columns swapped.
///
/// Column `j`'s entries are positions `col_offsets[j]..col_offsets[j + 1]` of
/// `row_indices` (their rows) and `values`. For the 3 × 4 matrix
///
/// ```text
/// [ 5  0  0  1 ]
/// [ 0  0  2  0 ]
/// [ 0  3  0  0 ]
/// ```
///
/// they are `col_offsets = [0, 1, 2, 3, 4]`, `row_indices = [0, 2, 1, 0]` and
/// `values = [5, 3, 2, 1]`. Entries that aren't stored are 0 unless
/// [`with_absent`](CscMatrix::with_absent) says they are missing.
///
/// Each of the three is borrowed or moved in, never copied. See
/// <https://predict.rs/data.html#sparse-features>.
#[derive(Debug, Clone)]
pub struct CscMatrix<'a, F: Float = f64> {
    cols: Compressed<'a, F>,
}

impl<'a, F: Float> CscMatrix<'a, F> {
    /// A CSC matrix with `n_rows` rows, its structure checked. `col_offsets` has
    /// one entry per column plus one, so a matrix with no columns is `[0]`.
    /// `n_rows` is given because rows with no stored entries leave no trace in
    /// `row_indices`.
    ///
    /// The values are checked by the dataset's `validate_*` methods.
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// - `Malformed` with `context = "col_offsets"` if it is empty, doesn't start
    ///   at 0, or decreases.
    /// - `ShapeMismatch` with `context = "row_indices"` or `"values"` if either
    ///   doesn't have one entry per stored value, the last column offset.
    /// - `Malformed` with `context = "row_indices"` at the first column with a
    ///   row `>= n_rows`, or rows not strictly increasing (out of order or
    ///   repeated).
    pub fn new(
        n_rows: usize,
        col_offsets: impl Into<Cow<'a, [usize]>>,
        row_indices: impl Into<Cow<'a, [usize]>>,
        values: impl Into<Cow<'a, [F]>>,
    ) -> Result<Self> {
        let cols = Compressed::new(
            &NAMES,
            n_rows,
            col_offsets.into(),
            row_indices.into(),
            values.into(),
        )?;
        Ok(CscMatrix { cols })
    }

    /// Set what entries that aren't stored mean: [`Absent::Zero`] (the default)
    /// or [`Absent::Missing`].
    pub fn with_absent(mut self, absent: Absent) -> Self {
        self.cols.absent = absent;
        self
    }

    pub fn n_rows(&self) -> usize {
        self.cols.lane_len()
    }

    pub fn n_cols(&self) -> usize {
        self.cols.n_lanes()
    }

    /// The number of stored entries.
    pub fn n_stored(&self) -> usize {
        self.cols.values().len()
    }

    pub fn col_offsets(&self) -> &[usize] {
        self.cols.offsets()
    }

    pub fn row_indices(&self) -> &[usize] {
        self.cols.indices()
    }

    pub fn values(&self) -> &[F] {
        self.cols.values()
    }

    pub fn absent(&self) -> Absent {
        self.cols.absent
    }

    /// `NonFinite` at the first entry, in column order, that `is_rejected` picks
    /// out, its column counted from `col_offset`. With [`Absent::Missing`],
    /// entries that aren't stored are checked as NaN.
    pub(super) fn reject_first(
        &self,
        context: &'static str,
        col_offset: usize,
        is_rejected: impl Fn(F) -> bool,
    ) -> Result<()> {
        match self.cols.first_rejected(is_rejected) {
            Some((col, row, value)) => Err(PredictError::NonFinite {
                context,
                row,
                col: col_offset + col,
                value: value.into(),
            }),
            None => Ok(()),
        }
    }

    /// Every stored entry as `(row, col, value)`, column by column.
    pub(super) fn for_each_stored(&self, mut visit: impl FnMut(usize, usize, F)) {
        self.cols
            .for_each_stored(|col, row, value| visit(row, col, value));
    }

    /// The same matrix with every entry written out: absent ones as 0 or NaN.
    /// Column-major, as it is stored.
    pub(super) fn to_dense(&self) -> Array2<F> {
        self.cols.to_dense_by_lanes().reversed_axes()
    }
}
