//! Targets `y` attached to a dataset, checked when they are attached.

use ndarray::{ArrayView, ArrayView2, Axis, Dimension, Ix1, Ix2};

use std::ops::Range;

use super::CovariantCowArray;
use super::select::select_array;

use crate::Result;
use crate::error::reject_first;

mod sealed {
    use ndarray::{ArrayView, ArrayView2};

    use crate::Result;

    pub trait Sealed {
        /// The number of rows of targets, or `None` without targets.
        fn n_rows(&self) -> Option<usize>;
    }

    /// The value check behind [`Target`](super::Target). It lives in the private
    /// module so users can't call it; `with_y` is the only way to run it.
    pub trait CheckTarget: Sized {
        fn check(y: ArrayView2<'_, Self>) -> Result<()>;
    }

    /// How [`TargetDim`](super::TargetDim) presents targets as rows × outputs, so
    /// one check serves both shapes.
    pub trait AsOutputMatrix: ndarray::RemoveAxis {
        fn as_output_matrix<T>(y: ArrayView<'_, T, Self>) -> ArrayView2<'_, T>;
    }
}

/// A target element type, and what [`with_y`](super::Dataset::with_y) checks
/// about its values beyond the shape.
///
/// - `f32` and `f64` are regression targets: every value must be finite.
/// - `usize` is a class index for classification, `0` to `n_classes - 1`: a
///   `usize` can't be NaN, so there is nothing more to check.
///
/// The trait is sealed: estimators are written for exactly these types. Any
/// other element type doesn't compile, so encode class names or signed integers
/// as `usize` indices first:
///
/// ```compile_fail
/// use ndarray::array;
/// use predict::Dataset;
///
/// let (x, y) = (array![[1.0]], array![0_i64]);
/// let data = Dataset::new(&x).with_y(&y); // error: `i64` is not a `Target`
/// ```
pub trait Target: sealed::CheckTarget + Copy + Send + Sync + 'static {}

impl sealed::CheckTarget for f32 {
    fn check(y: ArrayView2<'_, f32>) -> Result<()> {
        reject_first("y", y, 0, |value| !value.is_finite())
    }
}

impl sealed::CheckTarget for f64 {
    fn check(y: ArrayView2<'_, f64>) -> Result<()> {
        reject_first("y", y, 0, |value| !value.is_finite())
    }
}

impl sealed::CheckTarget for usize {
    fn check(_y: ArrayView2<'_, usize>) -> Result<()> {
        Ok(())
    }
}

impl Target for f32 {}
impl Target for f64 {}
impl Target for usize {}

/// The shape of the targets: [`Ix1`](type@Ix1) for one target per row,
/// [`Ix2`](type@Ix2) for several (rows × outputs), and nothing else.
///
/// Several outputs cover multi-output regression (a float per output),
/// multi-label classification (a `0` / `1` class per label), and multi-class
/// multi-output (a class per output).
pub trait TargetDim: sealed::AsOutputMatrix {}

impl sealed::AsOutputMatrix for Ix1 {
    fn as_output_matrix<T>(y: ArrayView<'_, T, Ix1>) -> ArrayView2<'_, T> {
        y.insert_axis(Axis(1))
    }
}

impl sealed::AsOutputMatrix for Ix2 {
    fn as_output_matrix<T>(y: ArrayView<'_, T, Ix2>) -> ArrayView2<'_, T> {
        y
    }
}

impl TargetDim for Ix1 {}
impl TargetDim for Ix2 {}

/// `y` as a rows × outputs matrix, without copying: a 1-D `y` becomes one column.
pub(super) fn as_output_matrix<T, D: TargetDim>(y: ArrayView<'_, T, D>) -> ArrayView2<'_, T> {
    <D as sealed::AsOutputMatrix>::as_output_matrix(y)
}

/// Runs `T`'s value check, for `with_y`.
pub(super) fn check<T: Target>(y: ArrayView2<'_, T>) -> Result<()> {
    <T as sealed::CheckTarget>::check(y)
}

/// Whether a dataset has targets: [`NoTargets`] or [`Targets`], and nothing else.
pub trait TargetState: sealed::Sealed {
    /// The same targets, borrowed: what [`Dataset::view`](super::Dataset::view)
    /// holds.
    type Borrowed<'b>: TargetState
    where
        Self: 'b;

    #[doc(hidden)]
    fn borrowed(&self) -> Self::Borrowed<'_>;

    /// The targets of the chosen rows: borrowed when `range` is `Some` (the
    /// rows are that range), copied otherwise. Called only by
    /// `Dataset::select_rows`, after it checked every row.
    #[doc(hidden)]
    fn select_rows(&self, rows: &[usize], range: Option<Range<usize>>) -> Self::Borrowed<'_>;
}

/// No targets: data for an unsupervised estimator, or data to predict on.
#[derive(Debug, Clone, Copy)]
pub struct NoTargets;

/// Targets that passed [`with_y`](super::Dataset::with_y): one row per sample,
/// and valid for their type `T`.
///
/// `D` is [`Ix1`](type@Ix1) (the default) for one target per row, or
/// [`Ix2`](type@Ix2) for several.
/// The field is private, so checked targets can only come from `with_y`.
#[derive(Debug, Clone)]
pub struct Targets<'a, T, D: Dimension = Ix1> {
    pub(super) y: CovariantCowArray<'a, T, D>,
}

impl sealed::Sealed for NoTargets {
    fn n_rows(&self) -> Option<usize> {
        None
    }
}

impl<T, D: Dimension> sealed::Sealed for Targets<'_, T, D> {
    fn n_rows(&self) -> Option<usize> {
        Some(self.y.len_of(Axis(0)))
    }
}

impl TargetState for NoTargets {
    type Borrowed<'b> = NoTargets;

    fn borrowed(&self) -> NoTargets {
        NoTargets
    }

    fn select_rows(&self, _rows: &[usize], _range: Option<Range<usize>>) -> NoTargets {
        NoTargets
    }
}

impl<T: Target, D: TargetDim> TargetState for Targets<'_, T, D> {
    type Borrowed<'b>
        = Targets<'b, T, D>
    where
        Self: 'b;

    fn borrowed(&self) -> Targets<'_, T, D> {
        Targets {
            y: self.y.view().into(),
        }
    }

    fn select_rows(&self, rows: &[usize], range: Option<Range<usize>>) -> Targets<'_, T, D> {
        Targets {
            y: select_array(self.y.view(), rows, range),
        }
    }
}
