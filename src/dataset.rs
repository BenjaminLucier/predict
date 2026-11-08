//! Feature data tagged with what is known about its values.

mod state;

use std::marker::PhantomData;

use ndarray::{Array2, ArrayView2, CowArray, Ix2};

pub use state::{Finite, InfinityFree, NoInfinity, State, Unchecked};

use crate::error::reject_first;
use crate::{Float, Result};

/// Features `x`, one row per sample and one column per feature, tagged with a
/// state `S` that says what is known about the values.
///
/// - `F` is the float type, `f64` by default.
/// - `S` is [`Unchecked`] when the dataset is created; the `validate_*` methods
///   check the values once and move it to [`NoInfinity`] or [`Finite`].
///
/// A `Dataset` either borrows the caller's array without copying it
/// ([`new`](Dataset::new)) or owns its array ([`from_owned`](Dataset::from_owned),
/// and what transformers return). Both are the same type: `'a` is the lifetime of
/// the borrow, and an owned dataset is `Dataset<'static, ..>`.
///
/// States only move up. A `Finite` dataset has no `validate_finite`, so checking
/// twice doesn't compile:
///
/// ```compile_fail
/// use ndarray::array;
/// use predict::Dataset;
///
/// let x = array![[1.0, 2.0]];
/// let data = Dataset::new(x.view()).validate_finite().unwrap();
/// let again = data.validate_finite(); // error: no method on `Dataset<_, _, Finite>`
/// ```
#[derive(Debug, Clone)]
pub struct Dataset<'a, F: Float = f64, S: State = Unchecked> {
    x: CowArray<'a, F, Ix2>,
    state: PhantomData<S>,
}

impl<'a, F: Float> Dataset<'a, F, Unchecked> {
    /// Borrow the caller's array, without copying it.
    pub fn new(x: ArrayView2<'a, F>) -> Self {
        Dataset::with_state(x.into())
    }

    /// Take ownership of an array.
    pub fn from_owned(x: Array2<F>) -> Dataset<'static, F, Unchecked> {
        Dataset::with_state(x.into())
    }
}

impl<'a, F: Float> Dataset<'a, F, Unchecked> {
    /// Scan every value once and move to `NoInfinity`: NaN (a missing value) is
    /// kept, ±infinity is rejected.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_no_infinity"`, at the first infinity in
    /// memory order.
    pub fn validate_no_infinity(self) -> Result<Dataset<'a, F, NoInfinity>> {
        reject_first("validate_no_infinity", self.view(), |value| {
            value.is_infinite()
        })?;
        Ok(Dataset::with_state(self.x))
    }

    /// Scan every value once and move to `Finite`.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_finite"`, at the first NaN or infinity
    /// in memory order.
    pub fn validate_finite(self) -> Result<Dataset<'a, F, Finite>> {
        self.into_finite()
    }
}

impl<'a, F: Float> Dataset<'a, F, NoInfinity> {
    /// Scan every value once and move to `Finite`. Infinity is already ruled out,
    /// so this finds NaN.
    ///
    /// # Errors
    ///
    /// `NonFinite` with `context = "validate_finite"`, at the first NaN in memory
    /// order.
    pub fn validate_finite(self) -> Result<Dataset<'a, F, Finite>> {
        self.into_finite()
    }
}

impl<'a, F: Float, S: State> Dataset<'a, F, S> {
    // The shared body of the two public `validate_finite`s. They exist only on the
    // states that can still move up, so calling it on `Finite` data doesn't compile.
    fn into_finite(self) -> Result<Dataset<'a, F, Finite>> {
        reject_first("validate_finite", self.view(), |value| !value.is_finite())?;
        Ok(Dataset::with_state(self.x))
    }

    /// The one place a `Dataset` is assembled. It is private because the caller
    /// is responsible for `S` being true of `x`.
    fn with_state(x: CowArray<'a, F, Ix2>) -> Self {
        Dataset {
            x,
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
