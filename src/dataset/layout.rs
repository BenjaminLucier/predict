//! How a dataset's features are laid out in memory, as a type parameter: which
//! way the blocks run (orientation) and whether they store every entry
//! (density).

use std::fmt::Debug;
use std::marker::PhantomData;

use ndarray::{CowArray, Ix2};

use super::block::{Block, BlockView};
use super::select::Selection;
use super::{Absent, CscMatrix, CsrMatrix};
use crate::{Float, Result};
use sealed::Sealed;

pub(super) mod sealed {

    use crate::{Float, Result};

    // Seals the orientation and density markers.
    pub trait Marker {}

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
    }
}

/// Implemented by every [`Blocks`] type, and nothing else: how a dataset stores
/// its features.
///
/// The trait is sealed: code outside this crate can name it in bounds but cannot
/// implement it.
pub trait Layout<'a, F: Float>: sealed::Sealed<'a, F> {}

/// Which way every block runs: [`Rows`], [`Cols`] or [`MixedOrientation`].
///
/// Sealed. An estimator that reads one row at a time bounds on `Rows`, one that
/// reads one column at a time on `Cols`.
pub trait Orientation: sealed::Marker + Debug + Clone + Copy + Send + Sync + 'static {}

/// Whether every block stores every entry: [`Dense`], [`Sparse`] or
/// [`MixedDensity`].
///
/// Sealed. An estimator with only dense kernels bounds on `Dense`.
pub trait Density: sealed::Marker + Debug + Clone + Copy + Send + Sync + 'static {}

/// Every block stores its rows contiguously: row-major dense or CSR.
#[derive(Debug, Clone, Copy)]
pub struct Rows;

/// Every block stores its columns contiguously: column-major dense or CSC.
#[derive(Debug, Clone, Copy)]
pub struct Cols;

/// Some blocks run by rows and others by columns.
#[derive(Debug, Clone, Copy)]
pub struct MixedOrientation;

/// Every block stores every entry.
#[derive(Debug, Clone, Copy)]
pub struct Dense;

/// Every block stores only some entries.
#[derive(Debug, Clone, Copy)]
pub struct Sparse;

/// Some blocks are dense and others sparse.
#[derive(Debug, Clone, Copy)]
pub struct MixedDensity;

/// The marker for two sides placed side by side, one axis at a time: the same
/// marker on both sides stays, different ones become the mixed marker. Sealed;
/// this is what decides the type [`hstack`](crate::Dataset::hstack) returns.
pub trait Union<R>: sealed::Marker {
    /// The marker of the result.
    type Output;
}

impl sealed::Marker for Rows {}
impl sealed::Marker for Cols {}
impl sealed::Marker for MixedOrientation {}
impl Orientation for Rows {}
impl Orientation for Cols {}
impl Orientation for MixedOrientation {}

impl sealed::Marker for Dense {}
impl sealed::Marker for Sparse {}
impl sealed::Marker for MixedDensity {}
impl Density for Dense {}
impl Density for Sparse {}
impl Density for MixedDensity {}

// `left + right = output`, written out: the same marker stays, any two different
// ones give the mixed marker.
macro_rules! union {
    ($($left:ident + $right:ident = $output:ident),+ $(,)?) => {
        $(impl Union<$right> for $left {
            type Output = $output;
        })+
    };
}

union!(
    Rows + Rows = Rows,
    Rows + Cols = MixedOrientation,
    Rows + MixedOrientation = MixedOrientation,
    Cols + Rows = MixedOrientation,
    Cols + Cols = Cols,
    Cols + MixedOrientation = MixedOrientation,
    MixedOrientation + Rows = MixedOrientation,
    MixedOrientation + Cols = MixedOrientation,
    MixedOrientation + MixedOrientation = MixedOrientation,
);

union!(
    Dense + Dense = Dense,
    Dense + Sparse = MixedDensity,
    Dense + MixedDensity = MixedDensity,
    Sparse + Dense = MixedDensity,
    Sparse + Sparse = Sparse,
    Sparse + MixedDensity = MixedDensity,
    MixedDensity + Dense = MixedDensity,
    MixedDensity + Sparse = MixedDensity,
    MixedDensity + MixedDensity = MixedDensity,
);

/// Features stored as blocks of adjacent columns placed side by side, each with
/// one row per sample. `O` says which way every block runs and `D` whether
/// every block is dense; the number of blocks is not part of the type.
///
/// The four single-kind layouts have their own names: [`DenseRowMajor`],
/// [`DenseColMajor`], [`SparseRowMajor`] and [`SparseColMajor`].
#[derive(Debug, Clone)]
pub struct Blocks<'a, F: Float, O: Orientation, D: Density> {
    // Never empty: `Dataset::new` and `Dataset::from_csr` make one block, even
    // for a matrix with no columns, and `append` only adds blocks with the same
    // number of rows, so the row count is always the first block's.
    blocks: Vec<Block<'a, F>>,
    // A promise, like a dataset's state: every block is of a kind `O` and `D`
    // allow. Only this module's constructors and `append` set it.
    kinds: PhantomData<(O, D)>,
}

/// Dense blocks with each row contiguous in memory: what
/// [`Dataset::new`](crate::Dataset::new) makes.
pub type DenseRowMajor<'a, F = f64> = Blocks<'a, F, Rows, Dense>;

/// Dense blocks with each column contiguous in memory: what
/// [`Dataset::from_col_major`](crate::Dataset::from_col_major) makes.
pub type DenseColMajor<'a, F = f64> = Blocks<'a, F, Cols, Dense>;

/// Sparse blocks stored row by row (CSR): what
/// [`Dataset::from_csr`](crate::Dataset::from_csr) makes.
pub type SparseRowMajor<'a, F = f64> = Blocks<'a, F, Rows, Sparse>;

/// Sparse blocks stored column by column (CSC): what
/// [`Dataset::from_csc`](crate::Dataset::from_csc) makes.
pub type SparseColMajor<'a, F = f64> = Blocks<'a, F, Cols, Sparse>;

impl<'a, F: Float> DenseRowMajor<'a, F> {
    /// One dense block. The caller checks that its rows are contiguous.
    pub(super) fn dense_row_major(x: CowArray<'a, F, Ix2>) -> Self {
        Blocks::one(Block::DenseRowMajor(x))
    }
}

impl<'a, F: Float> SparseRowMajor<'a, F> {
    pub(super) fn csr(x: CsrMatrix<'a, F>) -> Self {
        Blocks::one(Block::Csr(x))
    }
}

impl<'a, F: Float> DenseColMajor<'a, F> {
    /// One dense block. The caller checks that its columns are contiguous.
    pub(super) fn dense_col_major(x: CowArray<'a, F, Ix2>) -> Self {
        Blocks::one(Block::DenseColMajor(x))
    }
}

impl<'a, F: Float> SparseColMajor<'a, F> {
    pub(super) fn csc(x: CscMatrix<'a, F>) -> Self {
        Blocks::one(Block::Csc(x))
    }
}

impl<'a, F: Float, O: Orientation, D: Density> Blocks<'a, F, O, D> {
    fn one(block: Block<'a, F>) -> Self {
        Blocks {
            blocks: vec![block],
            kinds: PhantomData,
        }
    }

    pub(super) fn into_blocks(self) -> Vec<Block<'a, F>> {
        self.blocks
    }

    /// The same blocks, each borrowing what it holds: a new list of headers,
    /// no values copied.
    pub(super) fn borrowed(&self) -> Blocks<'_, F, O, D> {
        Blocks {
            blocks: self.blocks.iter().map(Block::borrowed).collect(),
            kinds: PhantomData,
        }
    }

    /// The chosen rows of every block, each block keeping its kind.
    pub(super) fn select_rows(&self, selection: &Selection<'_>) -> Blocks<'_, F, O, D> {
        Blocks {
            blocks: self
                .blocks
                .iter()
                .map(|block| block.select_rows(selection))
                .collect(),
            kinds: PhantomData,
        }
    }

    /// Every block, borrowed, in column order.
    pub(super) fn views(&self) -> impl ExactSizeIterator<Item = BlockView<'_, F>> {
        self.blocks.iter().map(Block::view)
    }

    /// What the sparse blocks' entries that aren't stored mean, or `None` with
    /// no sparse block. Every sparse block agrees: `hstack` refuses to put
    /// different meanings side by side.
    pub(super) fn absent(&self) -> Option<Absent> {
        self.blocks.iter().find_map(Block::absent)
    }

    /// Every sparse block's entries that aren't stored given the meaning
    /// `absent`; dense blocks are kept as they are.
    pub(super) fn with_absent(self, absent: Absent) -> Self {
        Blocks {
            blocks: self
                .blocks
                .into_iter()
                .map(|block| block.with_absent(absent))
                .collect(),
            kinds: PhantomData,
        }
    }

    /// `right`'s blocks after these, with each marker the union of both sides'.
    pub(super) fn append<O2, D2>(
        mut self,
        right: Blocks<'a, F, O2, D2>,
    ) -> Blocks<'a, F, <O as Union<O2>>::Output, <D as Union<D2>>::Output>
    where
        O: Union<O2, Output: Orientation>,
        D: Union<D2, Output: Density>,
        O2: Orientation,
        D2: Density,
    {
        // Only a bug in this crate can break this: `hstack` returns an error
        // first.
        debug_assert_eq!(
            self.n_rows(),
            right.n_rows(),
            "blocks need the same number of rows"
        );
        self.blocks.extend(right.blocks);
        Blocks {
            blocks: self.blocks,
            kinds: PhantomData,
        }
    }
}

impl<'a, F: Float, O: Orientation, D: Density> sealed::Sealed<'a, F> for Blocks<'a, F, O, D> {
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
}

impl<'a, F: Float, O: Orientation, D: Density> Layout<'a, F> for Blocks<'a, F, O, D> {}
