//! Feature data tagged with what is known about its values, and optional
//! targets checked when they are attached.

mod state;
mod targets;

use std::marker::PhantomData;

use ndarray::{ArrayView, ArrayView2, CowArray, Ix2};

pub use state::{Finite, InfinityFree, NoInfinity, State, Unchecked};
pub use targets::{NoTargets, Target, TargetDim, TargetState, Targets};

use crate::error::{check_len, reject_first};
use crate::{Float, PredictError, Result};

/// Features `x`, one row per sample and one column per feature, tagged with a
/// state `S` that says what is known about the values, plus optional targets `y`.
///
/// - `F` is the float type, `f64` by default.
/// - `S` is [`Unchecked`] when the dataset is created; the `validate_*` methods
///   check the values once and move it to [`NoInfinity`] or [`Finite`].
/// - `Y` is [`NoTargets`] until [`with_y`](Dataset::with_y) attaches checked
///   [`Targets`].
///
/// `S` and `Y` are independent: `S` describes only the features, and targets
/// are checked when they are attached, so the `validate_*` methods keep them as
/// they are.
///
/// A `Dataset` either borrows the caller's arrays or owns them, decided by how
/// they are passed to [`new`](Dataset::new) and [`with_y`](Dataset::with_y);
/// neither copies. Transformers return owned datasets. Both are the same type:
/// `'a` is the lifetime of the borrows, shared by features and targets, and an
/// owned dataset can be `Dataset<'static, ..>`.
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
    x: CowArray<'a, F, Ix2>,
    targets: Y,
    state: PhantomData<S>,
}

impl<'a, F: Float> Dataset<'a, F, Unchecked, NoTargets> {
    /// Wrap a feature matrix, without copying it.
    ///
    /// Pass `&x` or `x.view()` to borrow the array, or `x` to move it in. A moved-in
    /// array is owned by the dataset, which then doesn't borrow anything, so it can
    /// be returned from a function as `Dataset<'static, ..>`.
    pub fn new(x: impl Into<CowArray<'a, F, Ix2>>) -> Self {
        Dataset::from_parts(x.into(), NoTargets)
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
    /// borrows (`'a: 'b` reads "`'a` outlives `'b`"). Owned arrays borrow nothing,
    /// so owned features with owned targets stay `'static`.
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
    pub fn with_y<'b, T: Target, D: TargetDim>(
        self,
        y: impl Into<CowArray<'b, T, D>>,
    ) -> Result<Dataset<'b, F, S, Targets<'b, T, D>>>
    where
        'a: 'b,
    {
        let y = y.into();
        let y_matrix = targets::as_output_matrix(y.view());
        check_len("y", self.n_samples(), y_matrix.nrows())?;
        let n_outputs = y_matrix.ncols();
        if n_outputs == 0 {
            return Err(PredictError::EmptyInput { context: "y" });
        }
        targets::check(y_matrix)?;
        Ok(Dataset::from_parts(self.x, Targets { y }))
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
        reject_first("validate_no_infinity", self.view(), |value| {
            value.is_infinite()
        })?;
        Ok(Dataset::from_parts(self.x, self.targets))
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
        reject_first("validate_finite", self.view(), |value| !value.is_finite())?;
        Ok(Dataset::from_parts(self.x, self.targets))
    }

    /// The one place a `Dataset` is assembled. `S` comes from the caller's return
    /// type, so this can produce any state without checking anything. It is
    /// private because the caller is responsible for `S` being true of `x` and
    /// `Y` of `targets`.
    fn from_parts(x: CowArray<'a, F, Ix2>, targets: Y) -> Self {
        Dataset {
            x,
            targets,
            state: PhantomData,
        }
    }

    /// The feature matrix, borrowed.
    pub fn view(&self) -> ArrayView2<'_, F> {
        self.x.view()
    }

    /// The number of rows.
    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    /// The number of columns.
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }
}

impl<'a, F: Float, S: State, T: Target, D: TargetDim> Dataset<'a, F, S, Targets<'a, T, D>> {
    /// The checked targets, borrowed: 1-D, or rows × outputs.
    pub fn y(&self) -> ArrayView<'_, T, D> {
        self.targets.y.view()
    }
}
