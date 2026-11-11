//! How a dataset stores its features: blocks of adjacent columns, side by side.

use ndarray::{ArrayView2, ArrayViewMut2, Ix2};

use super::{Absent, CovariantCowArray, CscMatrix, CsrMatrix};
use crate::error::reject_first;
use crate::{Float, Result};

/// One block of a dataset's features, borrowed: a run of adjacent columns, all
/// stored the same way, with one row per sample. What
/// [`Dataset::blocks`](crate::Dataset::blocks) hands out, without copying.
///
/// More kinds of block may be added, so match it with a `_ =>` arm.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum BlockView<'b, F: Float = f64> {
    /// A dense matrix with each row contiguous.
    DenseRowMajor(ArrayView2<'b, F>),
    /// A dense matrix with each column contiguous.
    DenseColMajor(ArrayView2<'b, F>),
    /// A sparse matrix stored row by row.
    Csr(&'b CsrMatrix<'b, F>),
    /// A sparse matrix stored column by column.
    Csc(&'b CscMatrix<'b, F>),
}

/// A run of adjacent columns, all stored the same way, with one row per sample.
#[derive(Debug, Clone)]
pub(super) enum Block<'a, F: Float> {
    /// A dense matrix with each row contiguous, borrowed or owned.
    DenseRowMajor(CovariantCowArray<'a, F, Ix2>),
    /// A dense matrix with each column contiguous, borrowed or owned.
    DenseColMajor(CovariantCowArray<'a, F, Ix2>),
    /// A sparse matrix, row by row.
    Csr(CsrMatrix<'a, F>),
    /// A sparse matrix, column by column.
    Csc(CscMatrix<'a, F>),
}

impl<F: Float> Block<'_, F> {
    pub(super) fn n_rows(&self) -> usize {
        match self {
            Block::DenseRowMajor(x) | Block::DenseColMajor(x) => x.nrows(),
            Block::Csr(m) => m.n_rows(),
            Block::Csc(m) => m.n_rows(),
        }
    }

    pub(super) fn n_cols(&self) -> usize {
        match self {
            Block::DenseRowMajor(x) | Block::DenseColMajor(x) => x.ncols(),
            Block::Csr(m) => m.n_cols(),
            Block::Csc(m) => m.n_cols(),
        }
    }

    /// The same block, borrowing everything it holds.
    pub(super) fn borrowed(&self) -> Block<'_, F> {
        match self {
            Block::DenseRowMajor(x) => Block::DenseRowMajor(x.view().into()),
            Block::DenseColMajor(x) => Block::DenseColMajor(x.view().into()),
            Block::Csr(m) => Block::Csr(m.view()),
            Block::Csc(m) => Block::Csc(m.view()),
        }
    }

    /// The block, borrowed.
    pub(super) fn view(&self) -> BlockView<'_, F> {
        match self {
            Block::DenseRowMajor(x) => BlockView::DenseRowMajor(x.view()),
            Block::DenseColMajor(x) => BlockView::DenseColMajor(x.view()),
            Block::Csr(m) => BlockView::Csr(m),
            Block::Csc(m) => BlockView::Csc(m),
        }
    }

    /// `NonFinite` at the first value in memory order that `is_rejected` picks
    /// out, its column counted from `col_offset`: the block's first column in the
    /// dataset.
    pub(super) fn reject_first(
        &self,
        context: &'static str,
        col_offset: usize,
        is_rejected: impl Fn(F) -> bool,
    ) -> Result<()> {
        match self {
            Block::DenseRowMajor(x) | Block::DenseColMajor(x) => {
                reject_first(context, x.view(), col_offset, is_rejected)
            }
            Block::Csr(m) => m.reject_first(context, col_offset, is_rejected),
            Block::Csc(m) => m.reject_first(context, col_offset, is_rejected),
        }
    }

    /// What an entry this block doesn't store means, or `None` for a dense
    /// block, which stores every entry.
    pub(super) fn absent(&self) -> Option<Absent> {
        match self {
            Block::DenseRowMajor(_) | Block::DenseColMajor(_) => None,
            Block::Csr(m) => Some(m.absent()),
            Block::Csc(m) => Some(m.absent()),
        }
    }

    /// The same block with its entries that aren't stored given a new meaning.
    /// Dense blocks are kept as they are.
    pub(super) fn with_absent(self, absent: Absent) -> Self {
        match self {
            Block::Csr(m) => Block::Csr(m.with_absent(absent)),
            Block::Csc(m) => Block::Csc(m.with_absent(absent)),
            dense => dense,
        }
    }

    /// Every value of the block written into `out`, which has its shape: absent
    /// entries as what they mean.
    pub(super) fn write_dense(&self, mut out: ArrayViewMut2<'_, F>) {
        match self {
            Block::DenseRowMajor(x) | Block::DenseColMajor(x) => out.assign(x),
            Block::Csr(m) => {
                out.fill(m.absent().value());
                m.for_each_stored(|row, col, value| out[[row, col]] = value);
            }
            Block::Csc(m) => {
                out.fill(m.absent().value());
                m.for_each_stored(|row, col, value| out[[row, col]] = value);
            }
        }
    }

    /// Every entry a sparse matrix meaning `absent` has to store, as `(row, col,
    /// value)`, in the block's own order: row by row for a row-major or CSR
    /// block, column by column for a column-major or CSC one. Either order
    /// visits each row's columns, and each column's rows, in increasing order.
    ///
    /// A sparse block with the same meaning gives its stored entries as they
    /// are. One with the other meaning is written out first: each entry it
    /// leaves out now stands for a value the new meaning has to store.
    pub(super) fn for_each_entry(&self, absent: Absent, visit: impl FnMut(usize, usize, F)) {
        match self {
            Block::DenseRowMajor(x) => dense_entries(x.view(), false, absent, visit),
            Block::DenseColMajor(x) => dense_entries(x.view(), true, absent, visit),
            Block::Csr(m) if m.absent() == absent => m.for_each_stored(visit),
            Block::Csc(m) if m.absent() == absent => m.for_each_stored(visit),
            Block::Csr(m) => dense_entries(m.to_dense().view(), false, absent, visit),
            Block::Csc(m) => {
                let dense = m.to_dense();
                dense_entries(dense.view(), true, absent, visit);
            }
        }
    }
}

/// The values of `x` that `absent` doesn't stand for, row by row, or column by
/// column if `by_cols`.
fn dense_entries<F: Float>(
    x: ArrayView2<'_, F>,
    by_cols: bool,
    absent: Absent,
    mut visit: impl FnMut(usize, usize, F),
) {
    let (n_rows, n_cols) = x.dim();
    let mut keep = |row, col| {
        let value = x[[row, col]];
        if !absent.stands_for(value) {
            visit(row, col, value);
        }
    };
    if by_cols {
        (0..n_cols).for_each(|col| (0..n_rows).for_each(|row| keep(row, col)));
    } else {
        (0..n_rows).for_each(|row| (0..n_cols).for_each(|col| keep(row, col)));
    }
}
