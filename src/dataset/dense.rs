//! Dense matrices whose memory order has been checked: each row contiguous
//! (row-major) or each column contiguous (column-major).

use ndarray::{ArrayView2, CowArray, Ix2};

use super::CovariantCowArray;
use crate::{Float, PredictError, Result};

/// A dense matrix with each row contiguous in memory, checked once when it is
/// made. Wrapped by [`Dataset::from_row_major`](crate::Dataset::from_row_major).
///
/// Arrays built with `array!`, `zeros` or `from_shape_vec` are row-major, and so
/// is a view of some of their columns: each row is still contiguous, with a gap
/// before the next, as BLAS's leading dimension allows.
#[derive(Debug, Clone)]
pub struct RowMajorMatrix<'a, F: Float = f64> {
    x: CovariantCowArray<'a, F, Ix2>,
}

impl<'a, F: Float> RowMajorMatrix<'a, F> {
    /// Wrap `x` without copying it, if each of its rows is contiguous: pass `&x`
    /// or a view to borrow it, or `x` to move it in.
    ///
    /// # Errors
    ///
    /// `Malformed` with `context = "x"` if a row isn't contiguous: a transposed
    /// view, a column-major array, or a view that skips columns.
    pub fn new(x: impl Into<CowArray<'a, F, Ix2>>) -> Result<Self> {
        let x = x.into();
        if !rows_contiguous(&x.view()) {
            return Err(PredictError::Malformed {
                context: "x",
                reason: "the rows are not contiguous in memory; wrap column-major data \
                         in `ColMajorMatrix`, or copy it into row-major order with \
                         `x.as_standard_layout()`"
                    .to_string(),
            });
        }
        Ok(RowMajorMatrix { x })
    }

    /// The matrix, borrowed.
    pub fn view(&self) -> ArrayView2<'_, F> {
        self.x.view()
    }

    pub(super) fn into_inner(self) -> CovariantCowArray<'a, F, Ix2> {
        self.x
    }
}

/// A dense matrix with each column contiguous in memory (Fortran order),
/// checked once when it is made. Wrapped by
/// [`Dataset::from_col_major`](crate::Dataset::from_col_major).
///
/// `Array2::zeros((n, m).f())` makes one, and so does `x.t()` of a row-major
/// array, though that is the transpose: its rows are `x`'s columns.
#[derive(Debug, Clone)]
pub struct ColMajorMatrix<'a, F: Float = f64> {
    x: CovariantCowArray<'a, F, Ix2>,
}

impl<'a, F: Float> ColMajorMatrix<'a, F> {
    /// Wrap `x` without copying it, if each of its columns is contiguous: pass
    /// `&x` or a view to borrow it, or `x` to move it in.
    ///
    /// # Errors
    ///
    /// `Malformed` with `context = "x"` if a column isn't contiguous: a
    /// row-major array, or a view that skips rows.
    pub fn new(x: impl Into<CowArray<'a, F, Ix2>>) -> Result<Self> {
        let x = x.into();
        if !rows_contiguous(&x.t()) {
            return Err(PredictError::Malformed {
                context: "x",
                reason: "the columns are not contiguous in memory; wrap row-major data \
                         in `RowMajorMatrix`, or copy it into column-major order"
                    .to_string(),
            });
        }
        Ok(ColMajorMatrix { x })
    }

    /// The matrix, borrowed.
    pub fn view(&self) -> ArrayView2<'_, F> {
        self.x.view()
    }

    pub(super) fn into_inner(self) -> CovariantCowArray<'a, F, Ix2> {
        self.x
    }
}

/// Whether every row of `x` is contiguous in memory, rows following each other
/// forwards: unit column stride, and a row stride of at least the row length.
/// Padded rows pass; transposed, strided, broadcast and reversed views don't.
/// A single row or column has nothing to step over in that direction. The
/// column-major check is this on the transpose.
fn rows_contiguous<F>(x: &ArrayView2<'_, F>) -> bool {
    let (n_rows, n_cols) = x.dim();
    let (row_stride, col_stride) = (x.strides()[0], x.strides()[1]);
    let row_length = isize::try_from(n_cols.max(1)).unwrap_or(isize::MAX);
    (n_cols <= 1 || col_stride == 1) && (n_rows <= 1 || row_stride >= row_length)
}
