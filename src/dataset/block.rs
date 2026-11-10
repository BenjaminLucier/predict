//! How a dataset stores its features: blocks of adjacent columns, side by side.

use ndarray::{CowArray, Ix2};

use super::{CovariantCowArray, CscMatrix, CsrMatrix};
use crate::error::reject_first;
use crate::{Float, Result};

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

    /// Borrowed for a dense block, written out for a sparse one.
    pub(super) fn to_dense(&self) -> CowArray<'_, F, Ix2> {
        match self {
            Block::DenseRowMajor(x) | Block::DenseColMajor(x) => x.view().into(),
            Block::Csr(m) => m.to_dense().into(),
            Block::Csc(m) => m.to_dense().into(),
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
}
