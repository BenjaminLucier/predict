//! Holding rows out to estimate how well a model does on data it hasn't seen:
//! train/test splits, K-fold splits and cross-validation. See
//! <https://predict.rs/selection.html>.
//!
//! Every splitter implements [`Splitter`] and makes [`Split`]s: row numbers,
//! not data. [`Split::select`] turns one split into a training and a test
//! dataset, and [`CrossValidation`] does that for each split in turn, fitting
//! and scoring an estimator on each.

mod cross_validation;
mod kfold;
mod train_test;

pub use cross_validation::CrossValidation;
pub use kfold::{KFold, StratifiedKFold};
pub use train_test::TrainTestSplit;

use ndarray::ArrayView1;

use crate::Result;
use crate::dataset::{Blocks, Density, Orientation, State, TargetState};
use crate::{Dataset, Float};

/// One partition of a dataset's rows into a training side and a test side,
/// as row numbers. Both sides are ascending in every split this crate makes.
///
/// The fields are public, so a split chosen by hand ("the last 20 rows are
/// the test set") is a struct literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// The number of rows of the dataset this split was made for, which
    /// [`select`](Split::select) checks. Not always `train.len() +
    /// test.len()`: a split may leave rows out.
    pub n_samples: usize,
    /// The rows to fit on.
    pub train: Vec<usize>,
    /// The rows to score on.
    pub test: Vec<usize>,
}

/// A training side and a test side, named so they can't be swapped by
/// position: what [`Split::select`] returns.
#[derive(Debug, Clone)]
pub struct TrainTest<T> {
    pub train: T,
    pub test: T,
}

impl Split {
    /// This split's rows of `data`, as two datasets with its layout, state,
    /// targets and weights, built with
    /// [`select_rows`](Dataset::select_rows): a side that is one ascending
    /// range of rows is borrowed, any other is copied once.
    ///
    /// # Panics
    ///
    /// If `data` doesn't have `n_samples` rows, or a row is out of range: the
    /// split was made for different data.
    #[expect(
        clippy::type_complexity,
        reason = "one dataset type, spelled out twice"
    )]
    pub fn select<'d, 'a, F: Float, O: Orientation, D: Density, S: State, Y: TargetState>(
        &self,
        data: &'d Dataset<'a, F, Blocks<'a, F, O, D>, S, Y>,
    ) -> TrainTest<Dataset<'d, F, Blocks<'d, F, O, D>, S, Y::Borrowed<'d>>> {
        assert_eq!(
            data.n_samples(),
            self.n_samples,
            "this split was made for {} rows, not {}",
            self.n_samples,
            data.n_samples()
        );
        TrainTest {
            train: data.select_rows(&self.train),
            test: data.select_rows(&self.test),
        }
    }

    /// The split testing on `test` and training on every other row, both
    /// ascending.
    fn holding_out(n_samples: usize, mut test: Vec<usize>) -> Self {
        test.sort_unstable();
        let mut is_test = vec![false; n_samples];
        for &row in &test {
            is_test[row] = true;
        }
        let train = (0..n_samples).filter(|&row| !is_test[row]).collect();
        Split {
            n_samples,
            train,
            test,
        }
    }
}

/// The class of each row, for splitters that keep class proportions, such as
/// [`StratifiedKFold`]. Usually the targets, `Classes(data.y())`.
#[derive(Debug, Clone, Copy)]
pub struct Classes<'a>(pub ArrayView1<'a, usize>);

/// A way to partition rows into training and test sides: one split or many.
///
/// `By` names what the splitter reads besides the number of rows: `()` for
/// none, [`Classes`] for one class per row. Passing the wrong input is a
/// compile error, so a splitter can't be handed an input it would ignore:
///
/// ```compile_fail
/// use ndarray::array;
/// use predict::selection::{Classes, KFold, Splitter};
///
/// let classes = array![0_usize, 0, 1, 1];
/// KFold::new().splits(4, Classes(classes.view())); // KFold takes no classes
/// ```
pub trait Splitter {
    /// The per-row inputs this splitter reads.
    type By<'a>;

    /// Every split of `n_samples` rows, made up front.
    ///
    /// # Errors
    ///
    /// `ShapeMismatch` if a per-row input doesn't have `n_samples` entries,
    /// and `InvalidParameter` if a setting can't split `n_samples` rows.
    fn splits(&self, n_samples: usize, by: Self::By<'_>) -> Result<Vec<Split>>;
}
