//! Metrics for regression: how far predicted values are from the true ones.

use ndarray::ArrayView1;

use super::{check_regression_inputs, check_weights, mean, weighted_mean};
use crate::{Float, PredictError, Result};

/// `y_true[i] - y_pred[i]` for each row, in `f64`.
fn residuals<'b, F: Float>(
    y_true: ArrayView1<'b, F>,
    y_pred: ArrayView1<'b, F>,
) -> impl Iterator<Item = f64> + 'b {
    y_true
        .into_iter()
        .zip(y_pred)
        .map(|(&truth, &pred)| truth.into() - pred.into())
}

/// Mean squared error: the average of `(y_true[i] - y_pred[i])²`.
///
/// `0` is perfect; larger is worse. Squaring makes large errors count far more
/// than small ones, and the result is in the targets' units squared. `f64`
/// inputs whose squared differences, or their sum, exceed what `f64` can hold
/// give infinity, as in scikit-learn; `f32` inputs never do.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn mse<F: Float>(y_true: ArrayView1<'_, F>, y_pred: ArrayView1<'_, F>) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    Ok(mean(residuals(y_true, y_pred).map(|r| r * r), y_true.len()))
}

/// [`mse`] with each row counted `sample_weights[i]` times:
/// `Σ wᵢ (y_true[i] - y_pred[i])² / Σ wᵢ`.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn mse_weighted<F: Float, W: Float>(
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    weighted_squared_error("mse_weighted", y_true, y_pred, sample_weights)
}

/// The weighted mean of squared residuals, with `context` naming the public
/// function in an `Undefined` error.
fn weighted_squared_error<F: Float, W: Float>(
    context: &'static str,
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    let squares = residuals(y_true, y_pred).map(|r| r * r);
    weighted_mean(context, squares, y_true.len(), sample_weights)
}

/// Root mean squared error: `√mse`, in the targets' own units.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn rmse<F: Float>(y_true: ArrayView1<'_, F>, y_pred: ArrayView1<'_, F>) -> Result<f64> {
    Ok(mse(y_true, y_pred)?.sqrt())
}

/// [`rmse`] with each row counted `sample_weights[i]` times: `√mse_weighted`.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn rmse_weighted<F: Float, W: Float>(
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    Ok(weighted_squared_error("rmse_weighted", y_true, y_pred, sample_weights)?.sqrt())
}

/// Mean absolute error: the average of `|y_true[i] - y_pred[i]|`.
///
/// `0` is perfect; larger is worse. Every unit of error counts the same, so
/// one outlier moves it far less than it moves [`mse`].
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn mae<F: Float>(y_true: ArrayView1<'_, F>, y_pred: ArrayView1<'_, F>) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    Ok(mean(residuals(y_true, y_pred).map(f64::abs), y_true.len()))
}

/// [`mae`] with each row counted `sample_weights[i]` times:
/// `Σ wᵢ |y_true[i] - y_pred[i]| / Σ wᵢ`.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn mae_weighted<F: Float, W: Float>(
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    let absolutes = residuals(y_true, y_pred).map(f64::abs);
    weighted_mean("mae_weighted", absolutes, y_true.len(), sample_weights)
}

/// The coefficient of determination: how much better the predictions are than
/// always predicting the mean of `y_true`.
///
/// `R² = 1 - SS_res / SS_tot`, where `SS_res = Σ (y_true[i] - y_pred[i])²` is
/// the model's squared error and `SS_tot = Σ (y_true[i] - mean)²` that of the
/// mean. `1` is perfect, `0` is no better than the mean, and it is negative
/// when worse, with no lower bound. Unlike [`mse`] it has no units, so it
/// compares across problems. Predictions so bad that the ratio exceeds what
/// `f64` can hold give negative infinity, the way [`mse`] gives infinity.
///
/// # Errors
///
/// The input checks in the [module documentation](super), then `Undefined`
/// when `SS_tot` is zero, so there is no variance to explain: `y_true` is
/// constant, which includes a single row. That happens in small
/// cross-validation folds; scikit-learn returns `1.0` or `0.0` there instead
/// (`NaN` for one row). Also `Undefined` when `SS_tot` itself is outside the
/// range of `f64` (too large, or so small it rounds to zero).
pub fn r2_score<F: Float>(y_true: ArrayView1<'_, F>, y_pred: ArrayView1<'_, F>) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    let weights = std::iter::repeat_n(1.0, y_true.len());
    r2("r2_score", y_true, y_pred, weights)
}

/// [`r2_score`] with each row counted `sample_weights[i]` times, in both sums
/// and in the mean. Rows of weight zero are left out, so `y_true` counts as
/// constant when every row of positive weight has the same value.
///
/// # Errors
///
/// As for [`r2_score`], and the weight checks in the
/// [module documentation](super).
pub fn r2_score_weighted<F: Float, W: Float>(
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    check_regression_inputs(y_true, y_pred)?;
    check_weights("r2_score_weighted", y_true.len(), sample_weights)?;
    let weights = sample_weights.iter().map(|&w| w.into());
    r2("r2_score_weighted", y_true, y_pred, weights)
}

/// R² on checked inputs, with one weight per row.
fn r2<F: Float>(
    context: &'static str,
    y_true: ArrayView1<'_, F>,
    y_pred: ArrayView1<'_, F>,
    weights: impl Iterator<Item = f64> + Clone,
) -> Result<f64> {
    let undefined = |reason: &str| PredictError::Undefined {
        context,
        reason: reason.to_string(),
    };
    // Compared directly: rounding in the mean can leave a constant's SS_tot
    // slightly above zero.
    let mut counted = y_true.iter().zip(weights.clone()).filter(|&(_, w)| w > 0.0);
    let first = counted.next().map(|(&y, _)| y);
    if counted.all(|(&y, _)| Some(y) == first) {
        return Err(undefined(
            "y_true is constant, so there is no variance to explain",
        ));
    }

    let (sum_w, sum_wy) = y_true
        .iter()
        .zip(weights.clone())
        .fold((0.0, 0.0), |(sw, swy), (&y, w)| {
            (sw + w, swy + w * y.into())
        });
    let mean = sum_wy / sum_w;
    let ss_tot: f64 = y_true
        .iter()
        .zip(weights.clone())
        .map(|(&y, w)| w * (y.into() - mean).powi(2))
        .sum();
    if !(ss_tot > 0.0 && ss_tot.is_finite()) {
        return Err(undefined(
            "the total sum of squares is outside the range of f64",
        ));
    }
    let ss_res: f64 = residuals(y_true, y_pred)
        .zip(weights)
        .map(|(r, w)| w * r * r)
        .sum();
    Ok(1.0 - ss_res / ss_tot)
}
