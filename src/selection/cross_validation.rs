//! Cross-validation: fit and score an estimator on every split.

use ndarray::Array1;

use super::{Splitter, TrainTest};
use crate::dataset::{Blocks, Density, Orientation, State, Target, Targets};
use crate::metrics::Metric;
use crate::{Dataset, Fit, Float, Predict, Result};

/// Estimates how well an estimator does on rows it wasn't fitted on: for each
/// split its splitter makes, fit on the training side, predict the test side,
/// and score the predictions with a metric.
///
/// The splitter is always given: which rows train and which test is visible
/// in the calling code. For classification, [`StratifiedKFold`] with
/// `Classes(data.y())` keeps each fold's class mix.
///
/// [`StratifiedKFold`]: super::StratifiedKFold
#[derive(Debug, Clone)]
pub struct CrossValidation<C> {
    splitter: C,
}

impl<C: Splitter> CrossValidation<C> {
    /// Cross-validation over the splits `splitter` makes.
    pub fn new(splitter: C) -> Self {
        CrossValidation { splitter }
    }

    /// One score per split, in split order: `metric` on the test side's
    /// targets and the predictions for it.
    ///
    /// The splits are fitted one after another, and each split's rows are
    /// selected only when it is fitted and dropped after it is scored, so at
    /// most one split's copy exists at a time. With sample weights, the fit
    /// uses the training side's and the metric the test side's, so the score
    /// measures what the fit optimises.
    ///
    /// A plain mean of the scores weighs each split equally, whatever its
    /// size: K-fold sizes differ by at most one row.
    ///
    /// # Errors
    ///
    /// The first error of the splitter, then of fitting, predicting or
    /// scoring, in split order: a failed split has no score, not a NaN.
    pub fn score<'a, P, M, E, F, O, D, S, T>(
        &self,
        estimator: &E,
        data: &Dataset<'a, F, Blocks<'a, F, O, D>, S, Targets<'a, T>>,
        by: C::By<'_>,
        metric: &M,
    ) -> Result<Array1<f64>>
    where
        F: Float,
        O: Orientation,
        D: Density,
        S: State,
        T: Target,
        // Each split borrows `data` for as long as it is fitted, a lifetime
        // only this method knows, so the estimator must fit at any lifetime.
        E: for<'b> Fit<Dataset<'b, F, Blocks<'b, F, O, D>, S, Targets<'b, T>>, Fitted = P>,
        P: for<'b> Predict<
                Dataset<'b, F, Blocks<'b, F, O, D>, S, Targets<'b, T>>,
                Output = Array1<T>,
            >,
        M: Metric<T, F> + ?Sized,
    {
        let splits = self.splitter.splits(data.n_samples(), by)?;
        let mut scores = Vec::with_capacity(splits.len());
        for split in &splits {
            let TrainTest { train, test } = split.select(data);
            let fitted = estimator.fit(&train)?;
            // Before predicting: the training copy is the larger one.
            drop(train);
            let y_pred = fitted.predict(&test)?;
            scores.push(metric.compute(test.y(), y_pred.view(), test.sample_weights())?);
        }
        Ok(Array1::from(scores))
    }
}
