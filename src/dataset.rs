//! Feature data tagged with what is known about its values, with optional
//! targets and sample weights checked when they are attached.

mod block;
mod csr;
mod state;
mod targets;

use std::marker::PhantomData;

use ndarray::{ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, CowArray, CowRepr, Ix1, Ix2};

pub use csr::{Absent, CsrMatrix};
pub use state::{Finite, InfinityFree, NoInfinity, State, Unchecked};
pub use targets::{NoTargets, Target, TargetDim, TargetState, Targets};

use block::Block;

use crate::error::check_len;
use crate::{Float, PredictError, Result};

/// Exactly `CowArray<'a, A, D>`, spelled out, for struct fields.
///
/// ndarray 0.17 defines `CowArray` with its element type left to a default,
/// `<CowRepr<'a, A> as RawData>::Elem`. Rust treats any struct with such a field
/// as invariant over `'a`: a `Dataset<'static, ..>` could not be used where a
/// shorter-lived one is expected. Naming the element type `A` directly gives the
/// same type without that default, so `Dataset` and `Targets` are covariant, as
/// references are. The `datasets_shorten_like_references` test checks this at
/// compile time.
pub(crate) type CovariantCowArray<'a, A, D> = ArrayBase<CowRepr<'a, A>, D, A>;

/// Features `x`, one row per sample and one column per feature, tagged with a
/// state `S` that says what is known about the values, plus optional targets `y`
/// and sample weights.
///
/// - `F` is the float type, `f64` by default.
/// - `S` is [`Unchecked`] when the dataset is created; the `validate_*` methods
///   check the values once and move it to [`NoInfinity`] or [`Finite`].
/// - `Y` is [`NoTargets`] until [`with_y`](Dataset::with_y) attaches checked
///   [`Targets`].
///
/// Every row weighs 1 unless [`with_sample_weights`](Dataset::with_sample_weights)
/// sets a weight per row. Every estimator accepts weights, so they are a value in
/// the dataset rather than part of its type.
///
/// `S` describes only the features. Targets and weights are checked when they are
/// attached, so the `validate_*` methods keep them as they are.
///
/// A `Dataset` either borrows the caller's arrays or owns them, decided by how
/// they are passed to [`new`](Dataset::new), [`with_y`](Dataset::with_y) and
/// [`with_sample_weights`](Dataset::with_sample_weights); none of them copies. Transformers return owned datasets.
/// Both are the same type: `'a` is the lifetime of the borrows, shared by
/// features, targets and weights, and an owned dataset can be
/// `Dataset<'static, ..>`.
///
/// States only move up. A `Finite` dataset has no `validate_finite`, so checking
/// twice doesn't compile:
///
/// ```compile_fail
/// use ndarray::array;
/// use predict::Dataset;
///
/// let x = array![[1.0, 2.0]];
/// let data = Dataset::new(&x).validate_finite().unwrap();
/// let again = data.validate_finite(); // error: no method on `Dataset<_, _, Finite>`
/// ```
#[derive(Debug, Clone)]
pub struct Dataset<'a, F: Float = f64, S: State = Unchecked, Y: TargetState = NoTargets> {
    // The features, as blocks of columns side by side, each with `n_samples`
    // rows. Never empty: `new` makes one block, even for a matrix with no
    // columns, so the row count is always the first block's.
    blocks: Vec<Block<'a, F>>,
    targets: Y,
    // `None` means every row weighs 1, without storing a column of ones.
    sample_weights: Option<CovariantCowArray<'a, F, Ix1>>,
    state: PhantomData<S>,
}

impl<'a, F: Float> Dataset<'a, F, Unchecked, NoTargets> {
    /// Wrap a feature matrix, without copying it.
    ///
    /// Pass `&x` or `x.view()` to borrow the array, or `x` to move it in. A moved-in
    /// array is owned by the dataset, which then doesn't borrow anything, so it can
    /// be returned from a function as `Dataset<'static, ..>`.
    pub fn new(x: impl Into<CowArray<'a, F, Ix2>>) -> Self {
        Dataset::from_parts(vec![Block::Dense(x.into())], NoTargets, None)
    }

    /// Wrap a sparse feature matrix, without copying it: borrowed or owned as
    /// its arrays are. Its structure was checked when it was built; its values
    /// are checked by the `validate_*` methods, as for a dense one.
    pub fn from_csr(x: CsrMatrix<'a, F>) -> Self {
        Dataset::from_parts(vec![Block::Csr(x)], NoTargets, None)
    }
}

impl<'a, F: Float, S: State> Dataset<'a, F, S, NoTargets> {
    /// Attach targets, checking them as they are attached. Works in any feature
    /// state, without copying `y`: pass `&y` or `y.view()` to borrow it, or `y` to
    /// move it in.
    ///
    /// `y` is 1-D for one target per row, or 2-D (rows × outputs) for several;
    /// either way it has one row per sample.
    ///
    /// Features and targets then share one lifetime, the shorter of the two
    /// borrows: a longer-lived dataset shortens to the borrow of `y`, as a
    /// reference would. Owned arrays borrow nothing, so owned features with owned
    /// targets can be `'static`.
    ///
    /// Targets are attached once: a dataset with targets has no `with_y`, so
    /// attaching again doesn't compile:
    ///
    /// ```compile_fail
    /// use ndarray::array;
    /// use predict::Dataset;
    ///
    /// let (x, y) = (array![[1.0]], array![1.0]);
    /// let data = Dataset::new(&x).with_y(&y).unwrap();
    /// let again = data.with_y(&y); // error: no method on `Dataset<.., Targets<..>>`
    /// ```
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// - `ShapeMismatch` with `context = "y"` if `y` doesn't have one row per
    ///   sample.
    /// - `EmptyInput` with `context = "y"` if a 2-D `y` has no outputs.
    /// - `NonFinite` with `context = "y"` at the first NaN or infinity in float
    ///   targets, in memory order; `col` is the output (`0` for 1-D `y`).
    pub fn with_y<T: Target, D: TargetDim>(
        self,
        y: impl Into<CowArray<'a, T, D>>,
    ) -> Result<Dataset<'a, F, S, Targets<'a, T, D>>> {
        let y = y.into();
        let y_matrix = targets::as_output_matrix(y.view());
        check_len("y", self.n_samples(), y_matrix.nrows())?;
        let n_outputs = y_matrix.ncols();
        if n_outputs == 0 {
            return Err(PredictError::EmptyInput { context: "y" });
        }
        targets::check(y_matrix)?;
        Ok(Dataset::from_parts(
            self.blocks,
            Targets { y },
            self.sample_weights,
        ))
    }
}

impl<'a, F: Float, Y: TargetState> Dataset<'a, F, Unchecked, Y> {
    /// Scan every value once and move to `NoInfinity`: NaN (a missing value) is
    /// kept, ±infinity is rejected. Targets are kept as they are.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_no_infinity"`, at the first infinity in
    /// memory order.
    pub fn validate_no_infinity(self) -> Result<Dataset<'a, F, NoInfinity, Y>> {
        self.reject_first("validate_no_infinity", |value| value.is_infinite())?;
        Ok(Dataset::from_parts(
            self.blocks,
            self.targets,
            self.sample_weights,
        ))
    }

    /// Scan every value once and move to `Finite`. Targets are kept as they are.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_finite"`, at the first NaN or infinity
    /// in memory order.
    pub fn validate_finite(self) -> Result<Dataset<'a, F, Finite, Y>> {
        self.into_finite()
    }
}

impl<'a, F: Float, Y: TargetState> Dataset<'a, F, NoInfinity, Y> {
    /// Scan every value once and move to `Finite`. Infinity is already ruled out,
    /// so this finds NaN. Targets are kept as they are.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_finite"`, at the first NaN in memory
    /// order.
    pub fn validate_finite(self) -> Result<Dataset<'a, F, Finite, Y>> {
        self.into_finite()
    }
}

impl<'a, F: Float, S: State, Y: TargetState> Dataset<'a, F, S, Y> {
    // The shared body of the two public `validate_finite`s. They exist only on the
    // states that can still move up, so calling it on `Finite` data doesn't compile.
    fn into_finite(self) -> Result<Dataset<'a, F, Finite, Y>> {
        self.reject_first("validate_finite", |value| !value.is_finite())?;
        Ok(Dataset::from_parts(
            self.blocks,
            self.targets,
            self.sample_weights,
        ))
    }

    // `NonFinite` at the first value `is_rejected` picks out: blocks in column
    // order, then memory order within a block, with the column counted across
    // the whole dataset.
    fn reject_first(&self, context: &'static str, is_rejected: impl Fn(F) -> bool) -> Result<()> {
        let mut col_offset = 0;
        for block in &self.blocks {
            block.reject_first(context, col_offset, &is_rejected)?;
            col_offset += block.n_cols();
        }
        Ok(())
    }

    /// The one place a `Dataset` is assembled. `S` comes from the caller's return
    /// type, so this can produce any state without checking anything. It is
    /// private because the caller is responsible for `S` being true of `blocks`,
    /// `Y` of `targets`, `sample_weights` having been checked, and every block
    /// having the same number of rows.
    fn from_parts(
        blocks: Vec<Block<'a, F>>,
        targets: Y,
        sample_weights: Option<CovariantCowArray<'a, F, Ix1>>,
    ) -> Self {
        Dataset {
            blocks,
            targets,
            sample_weights,
            state: PhantomData,
        }
    }

    /// The features as one dense matrix: borrowed when they are stored as a
    /// single dense block, as a dataset made with [`new`](Dataset::new) is, and
    /// assembled into an owned matrix otherwise. Sparse entries that aren't
    /// stored are written out as 0, or as NaN when they mean missing.
    pub fn to_dense(&self) -> CowArray<'_, F, Ix2> {
        if let [block] = self.blocks.as_slice() {
            return block.to_dense();
        }
        let blocks: Vec<CowArray<'_, F, Ix2>> = self.blocks.iter().map(Block::to_dense).collect();
        let views: Vec<ArrayView2<'_, F>> = blocks.iter().map(|block| block.view()).collect();
        ndarray::concatenate(Axis(1), &views)
            .expect("every block has n_samples rows")
            .into()
    }

    /// The number of rows.
    pub fn n_samples(&self) -> usize {
        self.blocks[0].n_rows()
    }

    /// The number of columns, across all blocks.
    pub fn n_features(&self) -> usize {
        self.blocks.iter().map(Block::n_cols).sum()
    }

    /// The weight of each row, borrowed, or `None` when every row weighs 1.
    ///
    /// Estimators match on it: `None` takes the unweighted path, which gives the
    /// same result as a column of ones without multiplying by them.
    pub fn sample_weights(&self) -> Option<ArrayView1<'_, F>> {
        self.sample_weights.as_ref().map(|weights| weights.view())
    }
}

impl<'a, F: Float, S: State, Y: TargetState> Dataset<'a, F, S, Y> {
    /// Set a weight per row, replacing any set before. Works in any feature state,
    /// before or after [`with_y`](Dataset::with_y), without copying `weights`: pass
    /// `&weights` or `weights.view()` to borrow them, or `weights` to move them in.
    ///
    /// A weight scales how much a row contributes to what an estimator fits: for
    /// most estimators, weight 2 has the same effect as the row appearing twice,
    /// and weight 0 the same as leaving it out. Each estimator documents where it
    /// differs. Like targets, the weights share the dataset's lifetime.
    ///
    /// Whether any weight is left to learn from (a positive total) is checked by
    /// the estimator in `fit`, as for an empty dataset: a subset of the rows, such
    /// as a cross-validation fold, can weigh nothing even when the whole doesn't.
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// - `ShapeMismatch` with `context = "sample_weights"` if there isn't one
    ///   weight per sample.
    /// - At the first bad row: `NonFinite` with `context = "sample_weights"` for
    ///   a NaN or infinite weight, or `InvalidValue` for a negative one.
    pub fn with_sample_weights(self, weights: impl Into<CowArray<'a, F, Ix1>>) -> Result<Self> {
        let weights = checked_weights(self.n_samples(), weights.into())?;
        Ok(Dataset::from_parts(
            self.blocks,
            self.targets,
            Some(weights),
        ))
    }
}

/// `weights` if there is one per sample and each is finite and non-negative,
/// checked row by row. `-0.0` is accepted: it compares equal to `0.0`.
fn checked_weights<F: Float>(
    n_samples: usize,
    weights: CowArray<'_, F, Ix1>,
) -> Result<CowArray<'_, F, Ix1>> {
    check_len("sample_weights", n_samples, weights.len())?;
    for (row, &weight) in weights.indexed_iter() {
        if !weight.is_finite() {
            return Err(PredictError::NonFinite {
                context: "sample_weights",
                row,
                col: 0,
                value: weight.into(),
            });
        }
        if weight < F::zero() {
            return Err(PredictError::InvalidValue {
                context: "sample_weights",
                row,
                col: 0,
                value: weight.into(),
                reason: "must be non-negative".to_string(),
            });
        }
    }
    Ok(weights)
}

impl<'a, F: Float, S: State, T: Target, D: TargetDim> Dataset<'a, F, S, Targets<'a, T, D>> {
    /// The checked targets, borrowed: 1-D, or rows × outputs.
    pub fn y(&self) -> ArrayView<'_, T, D> {
        self.targets.y.view()
    }
}

// Only `new` creates a dataset so far, and it makes one block. These build
// several blocks directly, to check what depends on the column layout.
#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn side_by_side<'a>(blocks: &[&'a ndarray::Array2<f64>]) -> Dataset<'a> {
        let blocks = blocks.iter().map(|x| Block::Dense(x.view().into()));
        Dataset::from_parts(blocks.collect(), NoTargets, None)
    }

    #[test]
    fn several_blocks_read_as_one_matrix() {
        let (a, b) = (array![[1.0], [2.0]], array![[3.0, 4.0], [5.0, 6.0]]);
        let data = side_by_side(&[&a, &b]);
        assert_eq!((data.n_samples(), data.n_features()), (2, 3));

        let dense = data.to_dense();
        assert!(dense.is_owned());
        assert_eq!(dense, array![[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]);
    }

    #[test]
    fn a_bad_value_is_reported_at_its_column_in_the_dataset() {
        // The NaN is in column 1 of the second block, so column 2 of the dataset.
        let (a, b) = (array![[1.0], [2.0]], array![[3.0, 4.0], [5.0, f64::NAN]]);
        let err = side_by_side(&[&a, &b]).validate_finite().unwrap_err();
        assert!(matches!(
            err,
            PredictError::NonFinite { row: 1, col: 2, .. }
        ));

        // Blocks are scanned in column order: the first block's bad value wins.
        let a = array![[f64::INFINITY], [2.0]];
        let err = side_by_side(&[&a, &b]).validate_finite().unwrap_err();
        assert!(matches!(
            err,
            PredictError::NonFinite { row: 0, col: 0, .. }
        ));
    }
}
