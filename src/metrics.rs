//! Measures of how well predictions match the true targets.
//!
//! Every metric is a free function over 1-D array views, `y_true` and
//! `y_pred` (plus `sample_weights` for the `_weighted` forms), and checks its
//! inputs first: the output of `predict` is not validated (a fitted model can
//! produce infinity), and these functions take plain arrays, so nothing
//! upstream has checked them.
//!
//! - different lengths: `ShapeMismatch`, with `context = "y_pred"` or
//!   `"sample_weights"`;
//! - no rows: `EmptyInput`, with `context = "y_true"`;
//! - a NaN or infinity: `NonFinite`, with `context = "y_true"`, `"y_pred"` or
//!   `"sample_weights"`;
//! - a negative weight: `InvalidValue`, with `context = "sample_weights"`;
//! - weights that sum to zero, or to more than `f64` can hold: `Undefined`.
//!
//! Inputs are `f32` or `f64`; every result is `f64`, and the sums behind it are
//! taken in `f64` whatever the input type. A score is read and compared, never
//! fed back into the model's arithmetic, so the wider type costs nothing, and
//! it keeps `f32` sums accurate and free of overflow.

use ndarray::{ArrayView1, Axis, CowArray};

use crate::dataset::checked_weights;
use crate::error::{check_len, reject_first};
use crate::{Float, PredictError, Result};

mod classification;
mod metric;
mod regression;

pub use classification::{
    Average, ConfusionMatrix, accuracy, accuracy_weighted, f1_score, f1_score_weighted,
    fbeta_score, fbeta_score_weighted, precision, precision_weighted, recall, recall_weighted,
};
pub use metric::{Accuracy, Direction, F1, FBeta, MAE, MSE, Metric, Precision, R2, RMSE, Recall};
pub use regression::{
    mae, mae_weighted, mse, mse_weighted, r2_score, r2_score_weighted, rmse, rmse_weighted,
};

/// `y_pred` must have one value per value of `y_true`, and there must be at
/// least one.
fn check_lengths<A, B>(y_true: ArrayView1<'_, A>, y_pred: ArrayView1<'_, B>) -> Result<()> {
    check_len("y_pred", y_true.len(), y_pred.len())?;
    if y_true.is_empty() {
        return Err(PredictError::EmptyInput { context: "y_true" });
    }
    Ok(())
}

/// [`check_lengths`], and every value must be finite.
fn check_regression_inputs<F: Float>(
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
) -> Result<()> {
    check_lengths(y_true, y_pred)?;
    reject_first("y_true", y_true.insert_axis(Axis(1)), 0, |v| !v.is_finite())?;
    reject_first("y_pred", y_pred.insert_axis(Axis(1)), 0, |v| !v.is_finite())
}

/// The mean of `terms`, one per row.
fn mean(terms: impl Iterator<Item = f64>, n_rows: usize) -> f64 {
    terms.sum::<f64>() / n_rows as f64
}

/// Checks the weights the way `Dataset` does (one per row, finite,
/// non-negative), then that their total is positive and finite; `context`
/// names the public function in an `Undefined` error.
fn check_weights<W: Float>(
    context: &'static str,
    n_rows: usize,
    sample_weights: ArrayView1<'_, W>,
) -> Result<()> {
    checked_weights(n_rows, CowArray::from(sample_weights))?;
    let total: f64 = sample_weights.iter().map(|&w| w.into()).sum();
    if !(total > 0.0 && total.is_finite()) {
        return Err(PredictError::Undefined {
            context,
            reason: format!("the sample weights sum to {total}, not a positive finite number"),
        });
    }
    Ok(())
}

/// The mean of `terms` with each row counted `sample_weights[row]` times:
/// `Σ wᵢ·tᵢ / Σ wᵢ`, after [`check_weights`].
fn weighted_mean<W: Float>(
    context: &'static str,
    terms: impl Iterator<Item = f64>,
    n_rows: usize,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    check_weights(context, n_rows, sample_weights)?;
    let weights = sample_weights.iter().map(|&w| w.into());
    let (weighted, total) = terms
        .zip(weights)
        .fold((0.0, 0.0), |(sum, total), (t, w): (f64, f64)| {
            (sum + w * t, total + w)
        });
    Ok(weighted / total)
}
