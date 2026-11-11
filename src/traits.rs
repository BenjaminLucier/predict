//! The traits every estimator implements: checking its settings, fitting,
//! predicting and transforming. See <https://predict.rs/estimators.html>.
//!
//! An estimator is two types. The *config* holds the settings a user chooses
//! (`Ridge { alpha: 1.0 }`) and implements [`Fit`]; fitting returns a separate
//! *fitted* type holding what was learned, which implements [`Predict`] or
//! [`Transform`]. A config can't predict, and a fitted model can't be fitted
//! again by mistake.
//!
//! The traits are generic over their input `In`, usually a
//! [`Dataset`](crate::Dataset). An estimator implements them for exactly the
//! datasets it can use, named by their layout, state and targets, so passing
//! anything else is a compile error rather than a wrong model.

use crate::Result;

/// Checks an estimator's settings, before any data is read.
///
/// Every [`Fit`] calls it first, so a bad setting (a negative penalty, zero
/// clusters) is an `InvalidParameter` error in `fit`, never a panic or a wrong
/// model. Settings are not checked when they are set: `with_*` methods only
/// store them.
///
/// It is its own trait, not generic over the input, so the check is written once
/// per estimator however many inputs it fits. The default accepts everything,
/// for estimators without settings to check.
pub trait ValidateConfig {
    /// `Ok` if every setting is in range and consistent with the others.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` naming the first bad setting.
    fn validate_config(&self) -> Result<()> {
        Ok(())
    }
}

/// Learns from data: the config's side of an estimator.
///
/// `fit` borrows the data, because fitting only reads it: the same dataset can
/// be fitted again with other settings, or split into cross-validation folds,
/// without being copied. It returns the fitted model, a separate type.
///
/// # What the compiler rejects
///
/// An estimator implements `Fit` only for the datasets it can use. Take one
/// that fits finite, row-major dense features with float targets:
///
/// ```
/// # use ndarray::array;
/// # use predict::dataset::{DenseRowMajor, Finite, Targets};
/// # use predict::{Dataset, Fit, Result, ValidateConfig};
/// # struct Mean;
/// # impl ValidateConfig for Mean {}
/// # impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for Mean {
/// #     type Fitted = f64;
/// #     fn fit(&self, data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>) -> Result<f64> {
/// #         Ok(data.y().mean().unwrap_or(0.0))
/// #     }
/// # }
/// # let (x, y) = (array![[1.0], [2.0]], array![1.0, 3.0]);
/// let data = Dataset::new(&x).validate_finite()?.with_y(&y)?;
/// assert_eq!(Mean.fit(&data)?, 2.0);
/// # Ok::<(), predict::PredictError>(())
/// ```
///
/// Features that haven't been validated don't compile:
///
/// ```compile_fail
/// # use ndarray::array;
/// # use predict::dataset::{DenseRowMajor, Finite, Targets};
/// # use predict::{Dataset, Fit, Result, ValidateConfig};
/// # struct Mean;
/// # impl ValidateConfig for Mean {}
/// # impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for Mean {
/// #     type Fitted = f64;
/// #     fn fit(&self, data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>) -> Result<f64> {
/// #         Ok(data.y().mean().unwrap_or(0.0))
/// #     }
/// # }
/// # let (x, y) = (array![[1.0], [2.0]], array![1.0, 3.0]);
/// let data = Dataset::new(&x).with_y(&y)?;
/// Mean.fit(&data)?;
/// # Ok::<(), predict::PredictError>(())
/// ```
///
/// Nor do features without targets:
///
/// ```compile_fail
/// # use ndarray::array;
/// # use predict::dataset::{DenseRowMajor, Finite, Targets};
/// # use predict::{Dataset, Fit, Result, ValidateConfig};
/// # struct Mean;
/// # impl ValidateConfig for Mean {}
/// # impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for Mean {
/// #     type Fitted = f64;
/// #     fn fit(&self, data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>) -> Result<f64> {
/// #         Ok(data.y().mean().unwrap_or(0.0))
/// #     }
/// # }
/// # let (x, y) = (array![[1.0], [2.0]], array![1.0, 3.0]);
/// let data = Dataset::new(&x).validate_finite()?;
/// Mean.fit(&data)?;
/// # Ok::<(), predict::PredictError>(())
/// ```
///
/// Nor does another layout, until it is converted:
///
/// ```compile_fail
/// # use ndarray::array;
/// # use predict::dataset::{DenseRowMajor, Finite, Targets};
/// # use predict::{Dataset, Fit, Result, ValidateConfig};
/// # struct Mean;
/// # impl ValidateConfig for Mean {}
/// # impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for Mean {
/// #     type Fitted = f64;
/// #     fn fit(&self, data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>) -> Result<f64> {
/// #         Ok(data.y().mean().unwrap_or(0.0))
/// #     }
/// # }
/// # let (x, y) = (array![[1.0], [2.0]], array![1.0, 3.0]);
/// let data = Dataset::new(&x).into_csr().validate_finite()?.with_y(&y)?;
/// Mean.fit(&data)?;
/// # Ok::<(), predict::PredictError>(())
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be fitted on `{In}`",
    label = "`{Self}` doesn't implement `Fit` for this input",
    note = "check the input's state (`validate_finite()` or `validate_no_infinity()`), \
            its targets (`with_y()`), and its layout (convert with `into_dense_row_major()` \
            and the like): the estimator's docs list the datasets it fits"
)]
pub trait Fit<In>: ValidateConfig {
    /// What fitting learns: the model that predicts or transforms.
    type Fitted;

    /// Learn from `data`, after checking the settings.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` for a bad setting, and whatever else the estimator
    /// documents (an empty dataset, weights that add up to zero).
    fn fit(&self, data: &In) -> Result<Self::Fitted>;
}

/// Predicts targets for new rows: the fitted side of a supervised estimator.
///
/// `predict` accepts a dataset with or without targets: implementations are
/// generic over the targets and so can't read them, which rules out using the
/// answers to make the predictions. The output is owned.
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't predict on `{In}`",
    label = "`{Self}` doesn't implement `Predict` for this input",
    note = "check the input's state (`validate_finite()` or `validate_no_infinity()`) and \
            its layout (convert with `into_dense_row_major()` and the like): the model's \
            docs list the datasets it predicts on"
)]
pub trait Predict<In> {
    /// The predictions: one per row, such as an `Array1` of values or classes.
    type Output;

    /// Predict for every row of `x`.
    ///
    /// # Errors
    ///
    /// `ShapeMismatch` if `x` doesn't have the number of features the model was
    /// fitted on, and whatever else the model documents.
    fn predict(&self, x: &In) -> Result<Self::Output>;
}

/// Predicts how likely each class is: the fitted side of a probabilistic
/// classifier.
pub trait PredictProbability<In>: Predict<In> {
    /// The probabilities: rows × classes, each row adding up to 1, as an
    /// `Array2` of the model's float type.
    type Probabilities;

    /// The probability of each class for every row of `x`.
    ///
    /// # Errors
    ///
    /// As for [`predict`](Predict::predict).
    fn predict_probability(&self, x: &In) -> Result<Self::Probabilities>;

    /// The number of classes, the columns of the probabilities.
    fn n_classes(&self) -> usize;
}

/// Rewrites rows: the fitted side of a transformer, such as a scaler.
///
/// `transform` takes ownership of its input. Values the input owns are
/// rewritten in place; borrowed ones are copied once, when first written. The
/// output keeps the input's targets and weights, and its rows: a transformer
/// changes features, never which rows there are.
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't transform `{In}`",
    label = "`{Self}` doesn't implement `Transform` for this input",
    note = "check the input's state and layout: the transformer's docs list the datasets \
            it transforms"
)]
pub trait Transform<In> {
    /// The transformed data.
    type Output;

    /// Transform `x`, taking ownership of it.
    ///
    /// # Errors
    ///
    /// `ShapeMismatch` if `x` doesn't have the number of features the
    /// transformer was fitted on, and whatever else it documents.
    fn transform(&self, x: In) -> Result<Self::Output>;
}

/// Fits and transforms in one call, for transformers.
///
/// The default fits on a borrow of `data` and then transforms it, the same as
/// calling the two in turn. A transformer that can do both in one pass over the
/// data overrides it. Each transformer opts in with an empty impl, so the
/// shortcut exists only where it has been considered.
pub trait FitTransform<In>: Fit<In, Fitted: Transform<In>> {
    /// Fit on `data`, then transform it: the fitted transformer and the
    /// transformed data.
    ///
    /// # Errors
    ///
    /// Whatever [`fit`](Fit::fit) or [`transform`](Transform::transform)
    /// returns.
    #[expect(
        clippy::type_complexity,
        reason = "the pair is clearest spelled out; an alias would hide the projection"
    )]
    fn fit_transform(
        &self,
        data: In,
    ) -> Result<(Self::Fitted, <Self::Fitted as Transform<In>>::Output)> {
        let fitted = self.fit(&data)?;
        let transformed = fitted.transform(data)?;
        Ok((fitted, transformed))
    }
}
