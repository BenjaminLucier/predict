//! Compressed sparse row (CSR) matrices: only the stored entries, row by row.

use std::borrow::Cow;

use ndarray::Array2;

use crate::error::check_len;
use crate::{Float, PredictError, Result};

/// What an entry a sparse matrix doesn't store means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Absent {
    /// The entry is 0, as in scikit-learn and in the maths: the sparse matrix is
    /// the dense matrix with its zeros left out.
    #[default]
    Zero,
    /// The entry is missing, like a NaN, as in XGBoost. The matrix can then only
    /// be validated as `Finite` if it stores every entry.
    Missing,
}

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
/// (a `Vec`), never copied. See
/// <https://predict.rs/data.html#sparse-features>.
#[derive(Debug, Clone)]
pub struct CsrMatrix<'a, F: Float = f64> {
    n_cols: usize,
    row_offsets: Cow<'a, [usize]>,
    col_indices: Cow<'a, [usize]>,
    values: Cow<'a, [F]>,
    absent: Absent,
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
        let matrix = CsrMatrix {
            n_cols,
            row_offsets: row_offsets.into(),
            col_indices: col_indices.into(),
            values: values.into(),
            absent: Absent::Zero,
        };
        matrix.check_structure()?;
        Ok(matrix)
    }

    /// Set what entries that aren't stored mean: [`Absent::Zero`] (the default)
    /// or [`Absent::Missing`].
    pub fn with_absent(self, absent: Absent) -> Self {
        CsrMatrix { absent, ..self }
    }

    pub fn n_rows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The number of stored entries.
    pub fn n_stored(&self) -> usize {
        self.values.len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn absent(&self) -> Absent {
        self.absent
    }

    fn check_structure(&self) -> Result<()> {
        let offsets = &self.row_offsets[..];
        let Some(&first) = offsets.first() else {
            return malformed("row_offsets", "must have one entry per row plus one".into());
        };
        if first != 0 {
            return malformed("row_offsets", format!("must start at 0, got {first}"));
        }
        for (row, pair) in offsets.windows(2).enumerate() {
            if pair[1] < pair[0] {
                return malformed(
                    "row_offsets",
                    format!(
                        "must not decrease, got {} after {} at row {row}",
                        pair[1], pair[0]
                    ),
                );
            }
        }

        let n_stored = offsets[offsets.len() - 1];
        check_len("col_indices", n_stored, self.col_indices.len())?;
        check_len("values", n_stored, self.values.len())?;

        for row in 0..self.n_rows() {
            let (cols, _) = self.row(row);
            for (k, &col) in cols.iter().enumerate() {
                if col >= self.n_cols {
                    return malformed(
                        "col_indices",
                        format!(
                            "row {row}: column {col} is out of range for {} columns",
                            self.n_cols
                        ),
                    );
                }
                if k > 0 && col <= cols[k - 1] {
                    return malformed(
                        "col_indices",
                        format!(
                            "row {row}: column {col} does not come after column {}",
                            cols[k - 1]
                        ),
                    );
                }
            }
        }
        Ok(())
    }

    /// Row `row`'s stored columns and values.
    fn row(&self, row: usize) -> (&[usize], &[F]) {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        (&self.col_indices[range.clone()], &self.values[range])
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
        let check_absent = self.absent == Absent::Missing && is_rejected(F::nan());
        for row in 0..self.n_rows() {
            let (cols, values) = self.row(row);
            let stored = cols
                .iter()
                .zip(values)
                .find(|&(_, &value)| is_rejected(value))
                .map(|(&col, &value)| (col, value));
            let absent = check_absent
                .then(|| first_absent(cols, self.n_cols))
                .flatten()
                .map(|col| (col, F::nan()));
            if let Some((col, value)) = stored.into_iter().chain(absent).min_by_key(|&(col, _)| col)
            {
                return Err(PredictError::NonFinite {
                    context,
                    row,
                    col: col_offset + col,
                    value: value.into(),
                });
            }
        }
        Ok(())
    }

    /// The same matrix with every entry written out: absent ones as 0 or NaN.
    pub(super) fn to_dense(&self) -> Array2<F> {
        let fill = match self.absent {
            Absent::Zero => F::zero(),
            Absent::Missing => F::nan(),
        };
        let mut dense = Array2::from_elem((self.n_rows(), self.n_cols), fill);
        for row in 0..self.n_rows() {
            let (cols, values) = self.row(row);
            for (&col, &value) in cols.iter().zip(values) {
                dense[[row, col]] = value;
            }
        }
        dense
    }
}

/// The first column a row doesn't store. The stored columns are strictly
/// increasing, so it is the first position `k` whose column isn't `k`.
fn first_absent(cols: &[usize], n_cols: usize) -> Option<usize> {
    match cols.iter().enumerate().find(|&(k, &col)| col != k) {
        Some((k, _)) => Some(k),
        None => (cols.len() < n_cols).then_some(cols.len()),
    }
}

fn malformed(context: &'static str, reason: String) -> Result<()> {
    Err(PredictError::Malformed { context, reason })
}
