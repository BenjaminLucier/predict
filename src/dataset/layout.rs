//! How a dataset's features are laid out in memory, as a type parameter.

use ndarray::{ArrayView2, Axis, CowArray, Ix2};

use super::block::Block;
use crate::{Float, Result};
use sealed::Sealed;

pub(super) mod sealed {
    use ndarray::{CowArray, Ix2};

    use crate::{Float, Result};

    // The methods every layout gives `Dataset`. They live on this trait, which
    // can't be named outside the crate, so code outside can bound on `Layout`
    // but can neither implement it nor call these.
    pub trait Sealed<'a, F: Float> {
        fn n_rows(&self) -> usize;
        fn n_cols(&self) -> usize;
        // `NonFinite` at the first value `is_rejected` picks out, with its
        // column counted across the whole layout.
        fn reject_first(
            &self,
            context: &'static str,
            is_rejected: impl Fn(F) -> bool,
        ) -> Result<()>;
        fn to_dense(&self) -> CowArray<'_, F, Ix2>;
    }
}

/// Implemented by [`Blocks`], and nothing else: how a dataset stores its
/// features.
///
/// The trait is sealed: code outside this crate can name it in bounds but cannot
/// implement it.
pub trait Layout<'a, F: Float>: sealed::Sealed<'a, F> {}

/// Features stored as blocks of adjacent columns placed side by side, each dense
/// or sparse, with one row per sample.
#[derive(Debug, Clone)]
pub struct Blocks<'a, F: Float = f64> {
    // Never empty: `Dataset::new` and `Dataset::from_csr` make one block, even
    // for a matrix with no columns, and `append` only adds blocks with the same
    // number of rows, so the row count is always the first block's.
    blocks: Vec<Block<'a, F>>,
}

impl<'a, F: Float> Blocks<'a, F> {
    pub(super) fn one(block: Block<'a, F>) -> Self {
        Blocks {
            blocks: vec![block],
        }
    }

    /// `right`'s blocks after these.
    pub(super) fn append(mut self, right: Self) -> Self {
        // Only a bug in this crate can break this: `hstack` returns an error
        // first.
        debug_assert_eq!(
            self.n_rows(),
            right.n_rows(),
            "blocks need the same number of rows"
        );
        self.blocks.extend(right.blocks);
        self
    }
}

impl<'a, F: Float> sealed::Sealed<'a, F> for Blocks<'a, F> {
    fn n_rows(&self) -> usize {
        self.blocks[0].n_rows()
    }

    fn n_cols(&self) -> usize {
        self.blocks.iter().map(Block::n_cols).sum()
    }

    // Blocks in column order, then memory order within a block.
    fn reject_first(&self, context: &'static str, is_rejected: impl Fn(F) -> bool) -> Result<()> {
        let mut col_offset = 0;
        for block in &self.blocks {
            block.reject_first(context, col_offset, &is_rejected)?;
            col_offset += block.n_cols();
        }
        Ok(())
    }

    fn to_dense(&self) -> CowArray<'_, F, Ix2> {
        if let [block] = self.blocks.as_slice() {
            return block.to_dense();
        }
        let blocks: Vec<CowArray<'_, F, Ix2>> = self.blocks.iter().map(Block::to_dense).collect();
        let views: Vec<ArrayView2<'_, F>> = blocks.iter().map(|block| block.view()).collect();
        ndarray::concatenate(Axis(1), &views)
            .expect("every block has n_samples rows")
            .into()
    }
}

impl<'a, F: Float> Layout<'a, F> for Blocks<'a, F> {}
