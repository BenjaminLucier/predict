//! One split: a training side and a held-out test side.

use rand::seq::SliceRandom;

use super::{Split, Splitter};
use crate::error::require;
use crate::{Result, Seed};

/// Holds out `test_size` of the rows, chosen at random, as one test set: a
/// [`Splitter`] that makes one split.
///
/// The test side has `ceil(test_size * n_samples)` rows, rounded up as in
/// scikit-learn so it is never empty, and the training side the rest. Both
/// sides keep the rows' original relative order. Without shuffling, the last
/// rows are the test side, and each side is one range of rows, so
/// [`Split::select`] borrows both.
///
/// ```
/// use ndarray::array;
/// use predict::Dataset;
/// use predict::selection::{Splitter, TrainTest, TrainTestSplit};
///
/// let x = array![[1.0], [2.0], [3.0], [4.0]];
/// let data = Dataset::new(&x);
/// let split = TrainTestSplit::new().with_seed(42).splits(data.n_samples(), ())?.remove(0);
/// let TrainTest { train, test } = split.select(&data);
/// assert_eq!((train.n_samples(), test.n_samples()), (3, 1));
/// # Ok::<(), predict::PredictError>(())
/// ```
///
/// # Errors
///
/// `InvalidParameter` with `name = "test_size"` unless `0 < test_size < 1` and
/// at least one row is left to train on.
#[derive(Debug, Clone)]
pub struct TrainTestSplit {
    test_size: f64,
    /// `Some(seed)` shuffles with it, `None` keeps the rows in order.
    shuffle: Option<Seed>,
}

impl TrainTestSplit {
    /// A quarter of the rows as the test side, shuffled with a fresh seed.
    pub fn new() -> Self {
        TrainTestSplit {
            test_size: 0.25,
            shuffle: Some(Seed::Entropy),
        }
    }

    /// The fraction of rows to test on, strictly between 0 and 1.
    pub fn with_test_size(self, test_size: f64) -> Self {
        TrainTestSplit { test_size, ..self }
    }

    /// Shuffle with this seed, so the same rows are chosen every run.
    pub fn with_seed(self, seed: u64) -> Self {
        TrainTestSplit {
            shuffle: Some(Seed::Fixed(seed)),
            ..self
        }
    }

    /// Keep the rows in order: the last rows are the test side, as when
    /// testing on later data than the model was fitted on.
    pub fn without_shuffle(self) -> Self {
        TrainTestSplit {
            shuffle: None,
            ..self
        }
    }
}

impl Default for TrainTestSplit {
    fn default() -> Self {
        Self::new()
    }
}

impl Splitter for TrainTestSplit {
    type By<'a> = ();

    fn splits(&self, n_samples: usize, _by: ()) -> Result<Vec<Split>> {
        let test_size = self.test_size;
        require(
            test_size > 0.0 && test_size < 1.0,
            "test_size",
            &format!("must be strictly between 0 and 1, got {test_size}"),
        )?;
        // No epsilon, as scikit-learn: 0.07 * 100 is 7.000000000000001, so 8.
        let n_test = (test_size * n_samples as f64).ceil() as usize;
        require(
            n_test < n_samples,
            "test_size",
            &format!(
                "must leave a row to train on: {test_size} of {n_samples} rows tests {n_test}"
            ),
        )?;
        let split = match self.shuffle {
            Some(seed) => {
                let mut rows: Vec<usize> = (0..n_samples).collect();
                rows.shuffle(&mut seed.rng());
                rows.truncate(n_test);
                Split::holding_out(n_samples, rows)
            }
            None => {
                let n_train = n_samples - n_test;
                Split {
                    n_samples,
                    train: (0..n_train).collect(),
                    test: (n_train..n_samples).collect(),
                }
            }
        };
        Ok(vec![split])
    }
}
