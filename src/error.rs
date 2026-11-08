//! The crate-wide error type.

use thiserror::Error;

/// Shorthand for a `Result` whose error is [`PredictError`].
pub type Result<T, E = PredictError> = std::result::Result<T, E>;

/// Every fallible operation in `predict` returns this error.
///
/// It is `#[non_exhaustive]`, so match it with a `_ =>` arm: new variants can be
/// added without a breaking change. It does not implement `Clone` or `PartialEq`,
/// because later variants will wrap errors that don't (such as `std::io::Error`).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PredictError {
    /// A NaN or infinite value where finite values are required.
    ///
    /// `row` and `col` locate the first such value in row-major order; a 1-D input
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
