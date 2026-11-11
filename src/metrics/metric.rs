//! Metrics as values, for code that evaluates models: how to compute one, and
//! which way is better.

use ndarray::ArrayView1;

use super::{
    Average, accuracy, accuracy_weighted, f1_score, f1_score_weighted, fbeta_score,
    fbeta_score_weighted, mae, mae_weighted, mse, mse_weighted, precision, precision_weighted,
    r2_score, r2_score_weighted, recall, recall_weighted, rmse, rmse_weighted,
};
use crate::{Float, Result};

/// Whether a larger value of a metric means a better model.
///
/// scikit-learn instead makes every score "higher is better" by negating the
/// errors, which is why its users meet `neg_mean_squared_error`. Here an error
/// stays positive and says which way it points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

/// A metric as a value: what something that evaluates models (such as
/// cross-validation) is given to score each prediction with.
///
/// `T` is the type of the targets and predictions: `f32` or `f64` for the
/// regression metrics, `usize` classes for the classification ones. `W` is the
/// type of the sample weights. Each implementation calls the free function of
/// the same name, or its `_weighted` form when given weights; settings such as
/// `beta` are checked when the metric is computed.
///
/// The trait is open: implement it for your own type to score with a metric
/// of your own. It has no generic methods, so `dyn Metric<T, W>` works, for a
/// list of metrics of different types.
pub trait Metric<T, W> {
    /// Which way is better.
    fn direction(&self) -> Direction;

    /// The metric's value for these predictions, with each row counted
    /// `sample_weights[i]` times when given.
    fn compute(
        &self,
        y_true: ArrayView1<'_, T>,
        y_pred: ArrayView1<'_, T>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64>;
}

/// Mean squared error, [`mse`]. Lower is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MSE;

/// Root mean squared error, [`rmse`]. Lower is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RMSE;

/// Mean absolute error, [`mae`]. Lower is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MAE;

/// The coefficient of determination, [`r2_score`]. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct R2;

/// The fraction of correct classes, [`accuracy`]. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Accuracy;

/// [`precision`], combined over classes by `average`. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Precision {
    pub average: Average,
}

/// [`recall`], combined over classes by `average`. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recall {
    pub average: Average,
}

/// [`f1_score`], combined over classes by `average`. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F1 {
    pub average: Average,
}

/// [`fbeta_score`] with this `beta`, combined over classes by `average`.
/// Higher is better.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FBeta {
    pub beta: f64,
    pub average: Average,
}

impl<F: Float, W: Float> Metric<F, W> for MSE {
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, F>,
        y_pred: ArrayView1<'_, F>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => mse(y_true, y_pred),
            Some(weights) => mse_weighted(y_true, y_pred, weights),
        }
    }
}

impl<F: Float, W: Float> Metric<F, W> for RMSE {
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, F>,
        y_pred: ArrayView1<'_, F>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => rmse(y_true, y_pred),
            Some(weights) => rmse_weighted(y_true, y_pred, weights),
        }
    }
}

impl<F: Float, W: Float> Metric<F, W> for MAE {
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, F>,
        y_pred: ArrayView1<'_, F>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => mae(y_true, y_pred),
            Some(weights) => mae_weighted(y_true, y_pred, weights),
        }
    }
}

impl<F: Float, W: Float> Metric<F, W> for R2 {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, F>,
        y_pred: ArrayView1<'_, F>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => r2_score(y_true, y_pred),
            Some(weights) => r2_score_weighted(y_true, y_pred, weights),
        }
    }
}

impl<W: Float> Metric<usize, W> for Accuracy {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => accuracy(y_true, y_pred),
            Some(weights) => accuracy_weighted(y_true, y_pred, weights),
        }
    }
}

impl<W: Float> Metric<usize, W> for Precision {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => precision(y_true, y_pred, self.average),
            Some(weights) => precision_weighted(y_true, y_pred, self.average, weights),
        }
    }
}

impl<W: Float> Metric<usize, W> for Recall {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => recall(y_true, y_pred, self.average),
            Some(weights) => recall_weighted(y_true, y_pred, self.average, weights),
        }
    }
}

impl<W: Float> Metric<usize, W> for F1 {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        match sample_weights {
            None => f1_score(y_true, y_pred, self.average),
            Some(weights) => f1_score_weighted(y_true, y_pred, self.average, weights),
        }
    }
}

impl<W: Float> Metric<usize, W> for FBeta {
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Result<f64> {
        let (beta, average) = (self.beta, self.average);
        match sample_weights {
            None => fbeta_score(y_true, y_pred, beta, average),
            Some(weights) => fbeta_score_weighted(y_true, y_pred, beta, average, weights),
        }
    }
}
