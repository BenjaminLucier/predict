//! Feature data tagged with what is known about its values, with optional
//! targets and sample weights checked when they are attached.

mod block;
mod compressed;
mod convert;
mod csc;
mod csr;
mod dense;
mod layout;
mod state;
mod targets;

use std::marker::PhantomData;

use ndarray::{ArrayBase, ArrayView, ArrayView1, CowArray, CowRepr, Ix1, Ix2};

pub use block::BlockView;
pub use compressed::Absent;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dense::{ColMajorMatrix, RowMajorMatrix};
pub use layout::{
    Blocks, Cols, Dense, DenseColMajor, DenseRowMajor, Density, Layout, MixedDensity,
    MixedOrientation, Orientation, Rows, Sparse, SparseColMajor, SparseRowMajor, Union,
};
pub use state::{Finite, InfinityFree, NoInfinity, State, Unchecked};
pub use targets::{NoTargets, Target, TargetDim, TargetState, Targets};

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
/// - `L` is how the features are laid out in memory, a [`Blocks`] type:
///   [`DenseRowMajor`] by default, as [`new`](Dataset::new) makes.
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
/// let again = data.validate_finite(); // error: no method on `Dataset<_, _, _, Finite>`
/// ```
#[derive(Debug, Clone)]
pub struct Dataset<
    'a,
    F: Float = f64,
    L: Layout<'a, F> = DenseRowMajor<'a, F>,
    S: State = Unchecked,
    Y: TargetState = NoTargets,
> {
    features: L,
    targets: Y,
    // `None` means every row weighs 1, without storing a column of ones.
    sample_weights: Option<CovariantCowArray<'a, F, Ix1>>,
    state: PhantomData<S>,
}

impl<'a, F: Float> Dataset<'a, F, DenseRowMajor<'a, F>, Unchecked, NoTargets> {
    /// Wrap a row-major feature matrix, without copying it: the everyday
    /// constructor.
    ///
    /// Pass `&x` or `x.view()` to borrow the array, or `x` to move it in. A moved-in
    /// array is owned by the dataset, which then doesn't borrow anything, so it can
    /// be returned from a function as `Dataset<'static, ..>`.
    ///
    /// # Panics
    ///
    /// If a row of `x` isn't contiguous in memory. Arrays built with `array!`,
    /// `zeros` or `from_shape_vec` always are, and so is a view of some of their
    /// columns; a transposed view, a column-major (Fortran-order) array, or a
    /// view that skips columns isn't. For an array whose layout you don't
    /// control, check it with [`RowMajorMatrix::new`] and use
    /// [`from_row_major`](Dataset::from_row_major).
    pub fn new(x: impl Into<CowArray<'a, F, Ix2>>) -> Self {
        match RowMajorMatrix::new(x) {
            Ok(x) => Dataset::from_row_major(x),
            Err(err) => panic!("Dataset::new: {err}"),
        }
    }

    /// Wrap a dense matrix whose rows were checked to be contiguous, without
    /// copying it.
    pub fn from_row_major(x: RowMajorMatrix<'a, F>) -> Self {
        Dataset::from_parts(
            DenseRowMajor::dense_row_major(x.into_inner()),
            NoTargets,
            None,
        )
    }
}

impl<'a, F: Float> Dataset<'a, F, DenseColMajor<'a, F>, Unchecked, NoTargets> {
    /// Wrap a dense matrix whose columns were checked to be contiguous, without
    /// copying it.
    pub fn from_col_major(x: ColMajorMatrix<'a, F>) -> Self {
        Dataset::from_parts(
            DenseColMajor::dense_col_major(x.into_inner()),
            NoTargets,
            None,
        )
    }
}

impl<'a, F: Float> Dataset<'a, F, SparseRowMajor<'a, F>, Unchecked, NoTargets> {
    /// Wrap a sparse feature matrix stored row by row, without copying it:
    /// borrowed or owned as its arrays are. Its structure was checked when it
    /// was built; its values are checked by the `validate_*` methods, as for a
    /// dense one.
    pub fn from_csr(x: CsrMatrix<'a, F>) -> Self {
        Dataset::from_parts(SparseRowMajor::csr(x), NoTargets, None)
    }
}

impl<'a, F: Float> Dataset<'a, F, SparseColMajor<'a, F>, Unchecked, NoTargets> {
    /// Wrap a sparse feature matrix stored column by column, without copying it,
    /// as [`from_csr`](Dataset::from_csr) does for one stored row by row.
    pub fn from_csc(x: CscMatrix<'a, F>) -> Self {
        Dataset::from_parts(SparseColMajor::csc(x), NoTargets, None)
    }
}

impl<'a, F: Float, L: Layout<'a, F>, S: State> Dataset<'a, F, L, S, NoTargets> {
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
    ) -> Result<Dataset<'a, F, L, S, Targets<'a, T, D>>> {
        let y = y.into();
        let y_matrix = targets::as_output_matrix(y.view());
        check_len("y", self.n_samples(), y_matrix.nrows())?;
        let n_outputs = y_matrix.ncols();
        if n_outputs == 0 {
            return Err(PredictError::EmptyInput { context: "y" });
        }
        targets::check(y_matrix)?;
        Ok(Dataset::from_parts(
            self.features,
            Targets { y },
            self.sample_weights,
        ))
    }
}

impl<'a, F: Float, L: Layout<'a, F>, Y: TargetState> Dataset<'a, F, L, Unchecked, Y> {
    /// Scan every value once and move to `NoInfinity`: NaN (a missing value) is
    /// kept, ±infinity is rejected. Targets are kept as they are.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_no_infinity"`, at the first infinity in
    /// memory order.
    pub fn validate_no_infinity(self) -> Result<Dataset<'a, F, L, NoInfinity, Y>> {
        self.features
            .reject_first("validate_no_infinity", |value| value.is_infinite())?;
        Ok(Dataset::from_parts(
            self.features,
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
    pub fn validate_finite(self) -> Result<Dataset<'a, F, L, Finite, Y>> {
        self.into_finite()
    }
}

impl<'a, F: Float, L: Layout<'a, F>, Y: TargetState> Dataset<'a, F, L, NoInfinity, Y> {
    /// Scan every value once and move to `Finite`. Infinity is already ruled out,
    /// so this finds NaN. Targets are kept as they are.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_finite"`, at the first NaN in memory
    /// order.
    pub fn validate_finite(self) -> Result<Dataset<'a, F, L, Finite, Y>> {
        self.into_finite()
    }
}

impl<'a, F: Float, L: Layout<'a, F>, S: State, Y: TargetState> Dataset<'a, F, L, S, Y> {
    // The shared body of the two public `validate_finite`s. They exist only on the
    // states that can still move up, so calling it on `Finite` data doesn't compile.
    fn into_finite(self) -> Result<Dataset<'a, F, L, Finite, Y>> {
        self.features
            .reject_first("validate_finite", |value| !value.is_finite())?;
        Ok(Dataset::from_parts(
            self.features,
            self.targets,
            self.sample_weights,
        ))
    }

    /// The one place a `Dataset` is assembled. `S` comes from the caller's return
    /// type, so this can produce any state without checking anything. It is
    /// private because the caller is responsible for `S` being true of
    /// `features`, `Y` of `targets`, and `sample_weights` having been checked.
    fn from_parts(
        features: L,
        targets: Y,
        sample_weights: Option<CovariantCowArray<'a, F, Ix1>>,
    ) -> Self {
        Dataset {
            features,
            targets,
            sample_weights,
            state: PhantomData,
        }
    }

    /// The number of rows.
    pub fn n_samples(&self) -> usize {
        self.features.n_rows()
    }

    /// The number of columns.
    pub fn n_features(&self) -> usize {
        self.features.n_cols()
    }

    /// The weight of each row, borrowed, or `None` when every row weighs 1.
    ///
    /// Estimators match on it: `None` takes the unweighted path, which gives the
    /// same result as a column of ones without multiplying by them.
    pub fn sample_weights(&self) -> Option<ArrayView1<'_, F>> {
        self.sample_weights.as_ref().map(|weights| weights.view())
    }
}

impl<'a, F: Float, L: Layout<'a, F>, S: State, Y: TargetState> Dataset<'a, F, L, S, Y> {
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
            self.features,
            self.targets,
            Some(weights),
        ))
    }
}

impl<'a, F: Float, O: Orientation, D: Density, S: State, Y: TargetState>
    Dataset<'a, F, Blocks<'a, F, O, D>, S, Y>
{
    /// Place `right`'s features after these, as more columns of the same rows,
    /// without copying either: each block stays borrowed or owned as it was.
    /// Dense and sparse blocks can sit side by side; neither is converted.
    ///
    /// Both sides are in the same state, which the result keeps: stacking two
    /// `Finite` datasets gives a `Finite` one without scanning again. These
    /// targets and weights are kept. `right` has no targets, so stacking a
    /// dataset with targets on the right doesn't compile:
    ///
    /// ```compile_fail
    /// use ndarray::array;
    /// use predict::Dataset;
    ///
    /// let (x, y) = (array![[1.0]], array![1.0]);
    /// let right = Dataset::new(&x).with_y(&y).unwrap();
    /// let both = Dataset::new(&x).hstack(right); // error: expected `NoTargets`
    /// ```
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// - `ShapeMismatch` with `context = "hstack"` if `right` doesn't have the
    ///   same number of rows.
    /// - `Conflict` with `context = "hstack"` if `right` has sample weights:
    ///   attach weights to the stacked dataset instead.
    /// - `Conflict` with `context = "hstack"` if both sides have sparse blocks
    ///   whose entries that aren't stored mean different things (see
    ///   [`Absent`]): give them the same meaning first, with
    ///   [`with_absent`](Dataset::with_absent) or `CsrMatrix::with_absent`.
    #[expect(
        clippy::type_complexity,
        reason = "the result's layout is spelled out so it reads as the union rule"
    )]
    pub fn hstack<O2: Orientation, D2: Density>(
        self,
        right: Dataset<'a, F, Blocks<'a, F, O2, D2>, S>,
    ) -> Result<Dataset<'a, F, Blocks<'a, F, O::Output, D::Output>, S, Y>>
    where
        O: Union<O2, Output: Orientation>,
        D: Union<D2, Output: Density>,
    {
        check_len("hstack", self.n_samples(), right.n_samples())?;
        if right.sample_weights.is_some() {
            return Err(PredictError::Conflict {
                context: "hstack",
                reason: "the right side has sample weights; attach weights after stacking"
                    .to_string(),
            });
        }
        if let (Some(left), Some(right)) = (self.features.absent(), right.features.absent())
            && left != right
        {
            return Err(PredictError::Conflict {
                context: "hstack",
                reason: format!(
                    "the left side's sparse entries that aren't stored mean {left:?} and \
                     the right side's {right:?}; give both the same meaning with `with_absent`"
                ),
            });
        }
        Ok(Dataset::from_parts(
            self.features.append(right.features),
            self.targets,
            self.sample_weights,
        ))
    }

    /// Every block of features, borrowed, in column order: never copies. A dense
    /// block is an `ArrayView2`, a sparse one the `CsrMatrix` or `CscMatrix`
    /// itself. To see every value as one dense matrix, convert first, with
    /// [`into_dense_row_major`](Dataset::into_dense_row_major), whose result is
    /// one block.
    ///
    /// ```
    /// use ndarray::array;
    /// use predict::Dataset;
    /// use predict::dataset::BlockView;
    ///
    /// let x = array![[1.0, 2.0], [3.0, 4.0]];
    /// let data = Dataset::new(&x);
    /// assert_eq!(data.blocks().len(), 1);
    /// if let Some(BlockView::DenseRowMajor(view)) = data.blocks().next() {
    ///     assert_eq!(view.as_ptr(), x.as_ptr()); // the caller's array, not a copy
    /// }
    /// ```
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = BlockView<'_, F>> {
        self.features.views()
    }

    /// The features as one row-major dense matrix: absent sparse entries are
    /// written as what they mean, 0 or NaN. Moves the matrix, without copying,
    /// when it already is one row-major block; copies it into a new matrix
    /// otherwise. The state, targets and weights are kept: the values are the
    /// same, only stored differently.
    pub fn into_dense_row_major(self) -> Dataset<'a, F, DenseRowMajor<'a, F>, S, Y> {
        let n_rows = self.n_samples();
        let x = convert::into_dense(self.features.into_blocks(), n_rows, false);
        Dataset::from_parts(
            DenseRowMajor::dense_row_major(x),
            self.targets,
            self.sample_weights,
        )
    }

    /// The features as one column-major dense matrix, as
    /// [`into_dense_row_major`](Dataset::into_dense_row_major) does with each
    /// column contiguous instead.
    pub fn into_dense_col_major(self) -> Dataset<'a, F, DenseColMajor<'a, F>, S, Y> {
        let n_rows = self.n_samples();
        let x = convert::into_dense(self.features.into_blocks(), n_rows, true);
        Dataset::from_parts(
            DenseColMajor::dense_col_major(x),
            self.targets,
            self.sample_weights,
        )
    }

    /// The features as one CSR matrix, its entries that aren't stored meaning
    /// what the dataset's sparse blocks' do, or 0 without any: see
    /// [`into_csr_with`](Dataset::into_csr_with).
    pub fn into_csr(self) -> Dataset<'a, F, SparseRowMajor<'a, F>, S, Y> {
        let absent = self.features.absent().unwrap_or_default();
        self.into_csr_with(absent)
    }

    /// The features as one CSR matrix whose entries that aren't stored mean
    /// `absent`. Every value is kept: a value `absent` stands for (0, or NaN for
    /// [`Absent::Missing`]) is left out, and every other is stored, including
    /// those a sparse block left out with a different meaning, so converting a
    /// mostly-zero matrix to `Missing` stores its zeros.
    ///
    /// Moves the matrix, without copying, when it already is one CSR block with
    /// that meaning; builds a new one otherwise. The state, targets and weights
    /// are kept.
    pub fn into_csr_with(self, absent: Absent) -> Dataset<'a, F, SparseRowMajor<'a, F>, S, Y> {
        let n_rows = self.n_samples();
        let m = convert::into_csr(self.features.into_blocks(), n_rows, absent);
        Dataset::from_parts(SparseRowMajor::csr(m), self.targets, self.sample_weights)
    }

    /// The features as one CSC matrix: [`into_csr`](Dataset::into_csr) column
    /// by column.
    pub fn into_csc(self) -> Dataset<'a, F, SparseColMajor<'a, F>, S, Y> {
        let absent = self.features.absent().unwrap_or_default();
        self.into_csc_with(absent)
    }

    /// The features as one CSC matrix whose entries that aren't stored mean
    /// `absent`: [`into_csr_with`](Dataset::into_csr_with) column by column.
    pub fn into_csc_with(self, absent: Absent) -> Dataset<'a, F, SparseColMajor<'a, F>, S, Y> {
        let n_rows = self.n_samples();
        let m = convert::into_csc(self.features.into_blocks(), n_rows, absent);
        Dataset::from_parts(SparseColMajor::csc(m), self.targets, self.sample_weights)
    }
}

impl<'a, F: Float, O: Orientation, D: Density, Y: TargetState>
    Dataset<'a, F, Blocks<'a, F, O, D>, Unchecked, Y>
{
    /// Give every sparse block's entries that aren't stored the meaning
    /// `absent`, without copying anything: the stored entries stay, and what
    /// the rest stand for changes. Dense blocks are kept as they are.
    ///
    /// This changes the values, so it exists only before validation: a
    /// `Finite` dataset given [`Absent::Missing`] would hold NaN. To keep the
    /// values and change how they are stored instead, use
    /// [`into_csr_with`](Dataset::into_csr_with).
    ///
    /// ```compile_fail
    /// use predict::Dataset;
    /// use predict::dataset::{Absent, CsrMatrix};
    ///
    /// let m = CsrMatrix::new(2, vec![0, 1], vec![0], vec![1.0]).unwrap();
    /// let data = Dataset::from_csr(m).validate_finite().unwrap();
    /// let missing = data.with_absent(Absent::Missing); // error: only on `Unchecked`
    /// ```
    pub fn with_absent(self, absent: Absent) -> Self {
        Dataset::from_parts(
            self.features.with_absent(absent),
            self.targets,
            self.sample_weights,
        )
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

impl<'a, F: Float, L: Layout<'a, F>, S: State, T: Target, D: TargetDim>
    Dataset<'a, F, L, S, Targets<'a, T, D>>
{
    /// The checked targets, borrowed: 1-D, or rows × outputs.
    pub fn y(&self) -> ArrayView<'_, T, D> {
        self.targets.y.view()
    }
}
