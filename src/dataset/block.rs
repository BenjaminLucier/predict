//! How a dataset stores its features: blocks of adjacent columns, side by side.

use ndarray::Ix2;

use super::CovariantCowArray;
use crate::error::reject_first;
use crate::{Float, Result};

/// A run of adjacent columns, all stored the same way, with one row per sample.
#[derive(Debug, Clone)]
pub(super) enum Block<'a, F> {
    /// A dense matrix, borrowed or owned.
    Dense(CovariantCowArray<'a, F, Ix2>),
}

impl<F: Float> Block<'_, F> {
    pub(super) fn nrows(&self) -> usize {
        match self {
            Block::Dense(x) => x.nrows(),
        }
    }

    pub(super) fn ncols(&self) -> usize {
        match self {
            Block::Dense(x) => x.ncols(),
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
            Block::Dense(x) => reject_first(context, x.view(), col_offset, is_rejected),
        }
    }
}
