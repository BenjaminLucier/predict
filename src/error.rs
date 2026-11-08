//! The crate-wide error type, and the checks inside the crate that raise it.

use ndarray::{ArrayView2, FoldWhile, Zip};
use thiserror::Error;

use crate::Float;

/// Shorthand for a `Result` whose error is [`PredictError`].
pub type Result<T, E = PredictError> = std::result::Result<T, E>;

/// Every fallible operation in this crate returns this error.
///
/// It is `#[non_exhaustive]`, so match it with a `_ =>` arm: new variants can be
/// added without a breaking change. It does not implement `Clone` or `PartialEq`,
/// because later variants will wrap errors that don't (such as `std::io::Error`).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PredictError {
    /// A NaN or infinite value where finite values are required.
    ///
    /// `row` and `col` locate the first such value in memory order; with several,
    /// which one is reported depends on how the array is stored. A 1-D input
    /// reports `col = 0`.
    #[error("non-finite value {value} in {context} at row {row}, column {col}")]
    NonFinite {
        context: &'static str,
        row: usize,
        col: usize,
        value: f64,
    },

    /// A length or column count is not the one required, such as targets with a
    /// different number of rows than the features.
    #[error("{context}: expected {expected}, got {actual}")]
    ShapeMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },

    /// An input has no rows or no columns where at least one is required.
    #[error("{context}: empty input")]
    EmptyInput { context: &'static str },

    /// A setting is outside its valid range, or conflicts with the data.
    ///
    /// `reason` states the requirement, such as "must be at least 1".
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },
}

/// `NonFinite` for the first value in memory order that `is_rejected` picks out,
/// or `Ok` if there is none.
///
/// The scan follows the array's memory layout, so it reads memory sequentially
/// for row-major and column-major arrays alike, and stops at the first hit. With
/// several rejected values, which one is reported therefore depends on how the
/// array is stored. That is deterministic for a given array, but the traversal is
/// ndarray's `Zip` "efficient order", which ndarray doesn't formally fix, so an
/// ndarray upgrade could change which of several values is reported.
pub(crate) fn reject_first<F: Float>(
    context: &'static str,
    x: ArrayView2<'_, F>,
    is_rejected: impl Fn(F) -> bool,
) -> Result<()> {
    Zip::indexed(x)
        .fold_while(Ok(()), |_, (row, col), &value| {
            if is_rejected(value) {
                FoldWhile::Done(Err(PredictError::NonFinite {
                    context,
                    row,
                    col,
                    value: value.into(),
                }))
            } else {
                FoldWhile::Continue(Ok(()))
            }
        })
        .into_inner()
}

/// `ShapeMismatch` unless a length or column count is the expected one.
pub(crate) fn check_len(context: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(PredictError::ShapeMismatch {
        context,
        expected,
        actual,
    })
}
