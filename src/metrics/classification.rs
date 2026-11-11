//! Metrics for classification. Classes are `usize` indices, as in targets.

use ndarray::{Array2, ArrayView1};

use super::{check_lengths, check_weights, mean, weighted_mean};
use crate::{Float, PredictError, Result};

/// `1.0` for each row whose predicted class is the true one, else `0.0`.
fn hits<'b>(
    y_true: ArrayView1<'b, usize>,
    y_pred: ArrayView1<'b, usize>,
) -> impl Iterator<Item = f64> + 'b {
    y_true
        .into_iter()
        .zip(y_pred)
        .map(|(truth, pred)| if truth == pred { 1.0 } else { 0.0 })
}

/// The fraction of rows whose predicted class is the true one.
///
/// `1` is perfect. On imbalanced classes it flatters: always predicting the
/// majority class of a 95 / 5 split scores `0.95`.
///
/// # Errors
///
/// `ShapeMismatch` and `EmptyInput` as described in the
/// [module documentation](super). Classes need no other check.
pub fn accuracy(y_true: ArrayView1<'_, usize>, y_pred: ArrayView1<'_, usize>) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    Ok(mean(hits(y_true, y_pred), y_true.len()))
}

/// [`accuracy`] with each row counted `sample_weights[i]` times: the share of
/// the total weight on correctly classified rows.
///
/// # Errors
///
/// See the [module documentation](super) for the input checks.
pub fn accuracy_weighted<W: Float>(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let hits = hits(y_true, y_pred);
    weighted_mean("accuracy_weighted", hits, y_true.len(), sample_weights)
}

/// The classes that appear in `y_true` or `y_pred`, sorted, without repeats.
fn classes_of(y_true: ArrayView1<'_, usize>, y_pred: ArrayView1<'_, usize>) -> Vec<usize> {
    let mut classes: Vec<usize> = y_true.iter().chain(&y_pred).copied().collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

/// Where `class` sits in the sorted `classes`, which must contain it.
fn position(classes: &[usize], class: usize) -> usize {
    classes.partition_point(|&c| c < class)
}

/// A count of every (true class, predicted class) pair.
///
/// Rows are true classes and columns predicted ones, both in the order of
/// [`classes`](Self::classes): every class that appears in `y_true` or
/// `y_pred`, sorted. The diagonal holds the correct predictions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrix {
    matrix: Array2<usize>,
    classes: Vec<usize>,
}

impl ConfusionMatrix {
    /// Counts every `(y_true[i], y_pred[i])` pair.
    ///
    /// # Errors
    ///
    /// `ShapeMismatch` and `EmptyInput` as described in the
    /// [module documentation](super).
    pub fn from_predictions(
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
    ) -> Result<Self> {
        check_lengths(y_true, y_pred)?;
        let classes = classes_of(y_true, y_pred);
        let mut matrix = Array2::zeros((classes.len(), classes.len()));
        for (&truth, &pred) in y_true.iter().zip(&y_pred) {
            matrix[[position(&classes, truth), position(&classes, pred)]] += 1;
        }
        Ok(Self { matrix, classes })
    }

    /// The counts: rows are true classes, columns predicted ones.
    pub fn matrix(&self) -> &Array2<usize> {
        &self.matrix
    }

    /// The classes indexing both axes of [`matrix`](Self::matrix), sorted.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// The count of rows of `true_class` predicted as `predicted_class`, or
    /// `None` if either class appears in neither input.
    pub fn get(&self, true_class: usize, predicted_class: usize) -> Option<usize> {
        let row = self.classes.binary_search(&true_class).ok()?;
        let col = self.classes.binary_search(&predicted_class).ok()?;
        Some(self.matrix[[row, col]])
    }
}

/// How the per-class scores of [`precision`], [`recall`] and [`fbeta_score`]
/// combine into one number.
///
/// The classes are those that appear in `y_true` or `y_pred`. A class's
/// precision is `0.0` when it is never predicted, its recall is `0.0` when it
/// never occurs, and its F-beta is `0.0` when it has no true positives:
/// scikit-learn's default `zero_division`, without the warning. A class the
/// model never predicts really did get no value from it, and an error here
/// would stop cross-validation on any fold that misses a rare class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Average {
    /// The score of the `positive` class alone, for two-class problems.
    ///
    /// `InvalidParameter` when the inputs hold more than two classes, or two
    /// that don't include `positive`, as in scikit-learn. With one class that
    /// isn't `positive`, the score is `0.0`.
    Binary { positive: usize },
    /// The plain mean over classes: each class counts the same, however rare.
    Macro,
    /// One score from the true positives, false positives and false negatives
    /// summed over all classes. With one class per row, precision, recall and
    /// F-beta all equal accuracy.
    Micro,
    /// The mean over classes weighted by support, each class's share of the
    /// true rows. scikit-learn calls this `"weighted"`; the name here keeps it
    /// apart from the sample weights of the `_weighted` functions.
    BySupport,
}

/// One class's true positives, predicted rows and true rows, or the same
/// summed over classes for [`Average::Micro`]. Each is a total weight: a row
/// count when every row weighs 1.
#[derive(Debug, Default)]
struct Counts {
    true_positives: f64,
    predicted: f64,
    actual: f64,
}

impl Counts {
    fn precision(&self) -> f64 {
        ratio_or_zero(self.true_positives, self.predicted)
    }

    fn recall(&self) -> f64 {
        ratio_or_zero(self.true_positives, self.actual)
    }

    /// Written over precision and recall, both at most 1, so a large
    /// `beta_squared` can't overflow the way the same formula over the counts
    /// can.
    fn fbeta(&self, beta_squared: f64) -> f64 {
        if self.true_positives == 0.0 {
            return 0.0;
        }
        let (p, r) = (self.precision(), self.recall());
        (1.0 + beta_squared) * p * r / (beta_squared * p + r)
    }
}

fn ratio_or_zero(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// The confusion matrix as total weights, every row weighing 1 without
/// `sample_weights`. Private: the public [`ConfusionMatrix`] holds counts.
struct Tally {
    classes: Vec<usize>,
    matrix: Array2<f64>,
}

impl Tally {
    fn new<W: Float>(
        y_true: ArrayView1<'_, usize>,
        y_pred: ArrayView1<'_, usize>,
        sample_weights: Option<ArrayView1<'_, W>>,
    ) -> Self {
        let classes = classes_of(y_true, y_pred);
        let mut matrix = Array2::zeros((classes.len(), classes.len()));
        for (row, (&truth, &pred)) in y_true.iter().zip(&y_pred).enumerate() {
            let weight = sample_weights.map_or(1.0, |w| w[row].into());
            matrix[[position(&classes, truth), position(&classes, pred)]] += weight;
        }
        Self { classes, matrix }
    }

    fn counts(&self, index: usize) -> Counts {
        Counts {
            true_positives: self.matrix[[index, index]],
            predicted: self.matrix.column(index).sum(),
            actual: self.matrix.row(index).sum(),
        }
    }
}

/// Checks the inputs, applies `score` to the [`Counts`] that `average`
/// selects, and combines. `context` names the public function.
fn averaged<W: Float>(
    context: &'static str,
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
    sample_weights: Option<ArrayView1<'_, W>>,
    score: impl Fn(&Counts) -> f64,
) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    if let Some(weights) = sample_weights {
        check_weights(context, y_true.len(), weights)?;
    }
    let tally = Tally::new(y_true, y_pred, sample_weights);
    let n_classes = tally.classes.len();
    let per_class = || (0..n_classes).map(|index| tally.counts(index));
    Ok(match average {
        Average::Binary { positive } => {
            check_binary(&tally.classes, positive)?;
            match tally.classes.binary_search(&positive) {
                Ok(index) => score(&tally.counts(index)),
                Err(_) => score(&Counts::default()),
            }
        }
        Average::Macro => per_class().map(|c| score(&c)).sum::<f64>() / n_classes as f64,
        Average::Micro => {
            let total = tally.matrix.sum();
            score(&Counts {
                true_positives: tally.matrix.diag().sum(),
                predicted: total,
                actual: total,
            })
        }
        Average::BySupport => {
            let weighted: f64 = per_class().map(|c| c.actual * score(&c)).sum();
            weighted / tally.matrix.sum()
        }
    })
}

fn check_binary(classes: &[usize], positive: usize) -> Result<()> {
    if classes.len() > 2 {
        return Err(PredictError::InvalidParameter {
            name: "average",
            reason: format!("Binary needs at most two classes, got {classes:?}"),
        });
    }
    if classes.len() == 2 && !classes.contains(&positive) {
        return Err(PredictError::InvalidParameter {
            name: "average",
            reason: format!("positive class {positive} is not one of {classes:?}"),
        });
    }
    Ok(())
}

/// No sample weights, for the unweighted functions.
const UNWEIGHTED: Option<ArrayView1<'static, f64>> = None;

/// Of the rows predicted as a class, the fraction that truly are:
/// `TP / (TP + FP)`. See [`Average`] for how classes combine and for the
/// zero-division rule.
///
/// # Errors
///
/// `ShapeMismatch` and `EmptyInput` as described in the
/// [module documentation](super); `InvalidParameter` from [`Average::Binary`].
pub fn precision(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
) -> Result<f64> {
    averaged(
        "precision",
        y_true,
        y_pred,
        average,
        UNWEIGHTED,
        Counts::precision,
    )
}

/// [`precision`] with each row counted `sample_weights[i]` times.
///
/// # Errors
///
/// As for [`precision`], and the weight checks in the
/// [module documentation](super).
pub fn precision_weighted<W: Float>(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    let weights = Some(sample_weights);
    averaged(
        "precision_weighted",
        y_true,
        y_pred,
        average,
        weights,
        Counts::precision,
    )
}

/// Of the rows truly of a class, the fraction predicted as it:
/// `TP / (TP + FN)`. See [`Average`] for how classes combine and for the
/// zero-division rule.
///
/// # Errors
///
/// As for [`precision`].
pub fn recall(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
) -> Result<f64> {
    averaged(
        "recall",
        y_true,
        y_pred,
        average,
        UNWEIGHTED,
        Counts::recall,
    )
}

/// [`recall`] with each row counted `sample_weights[i]` times.
///
/// # Errors
///
/// As for [`precision_weighted`].
pub fn recall_weighted<W: Float>(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    let weights = Some(sample_weights);
    averaged(
        "recall_weighted",
        y_true,
        y_pred,
        average,
        weights,
        Counts::recall,
    )
}

/// `beta²`, or `InvalidParameter` unless `beta` is positive and its square
/// finite.
fn checked_beta_squared(beta: f64) -> Result<f64> {
    let beta_squared = beta * beta;
    if beta > 0.0 && beta_squared.is_finite() {
        return Ok(beta_squared);
    }
    Err(PredictError::InvalidParameter {
        name: "beta",
        reason: format!("must be positive with a finite square, got {beta}"),
    })
}

/// The F-beta score: a weighted harmonic mean of precision `P` and recall `R`,
/// `(1 + β²)·P·R / (β²·P + R)`.
///
/// `beta > 1` counts recall more, `beta < 1` precision more; `beta = 1` is
/// [`f1_score`]. A harmonic mean is pulled towards the smaller of the two, so
/// a model can't score well by maximising one at the expense of the other.
/// `Macro` and `BySupport` average the per-class scores, not the averaged
/// `P` and `R`.
///
/// # Errors
///
/// As for [`precision`]; `InvalidParameter` unless `beta` is positive with a
/// finite square.
pub fn fbeta_score(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    beta: f64,
    average: Average,
) -> Result<f64> {
    let beta_squared = checked_beta_squared(beta)?;
    averaged("fbeta_score", y_true, y_pred, average, UNWEIGHTED, |c| {
        c.fbeta(beta_squared)
    })
}

/// [`fbeta_score`] with each row counted `sample_weights[i]` times.
///
/// # Errors
///
/// As for [`fbeta_score`], and the weight checks in the
/// [module documentation](super).
pub fn fbeta_score_weighted<W: Float>(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    beta: f64,
    average: Average,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    let beta_squared = checked_beta_squared(beta)?;
    let weights = Some(sample_weights);
    averaged(
        "fbeta_score_weighted",
        y_true,
        y_pred,
        average,
        weights,
        |c| c.fbeta(beta_squared),
    )
}

/// The F1 score, the plain harmonic mean of precision and recall:
/// [`fbeta_score`] with `beta = 1`.
///
/// # Errors
///
/// As for [`precision`].
pub fn f1_score(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
) -> Result<f64> {
    averaged("f1_score", y_true, y_pred, average, UNWEIGHTED, |c| {
        c.fbeta(1.0)
    })
}

/// [`f1_score`] with each row counted `sample_weights[i]` times.
///
/// # Errors
///
/// As for [`precision_weighted`].
pub fn f1_score_weighted<W: Float>(
    y_true: ArrayView1<'_, usize>,
    y_pred: ArrayView1<'_, usize>,
    average: Average,
    sample_weights: ArrayView1<'_, W>,
) -> Result<f64> {
    let weights = Some(sample_weights);
    averaged("f1_score_weighted", y_true, y_pred, average, weights, |c| {
        c.fbeta(1.0)
    })
}
