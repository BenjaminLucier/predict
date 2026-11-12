use ndarray::{Array1, array};
use predict::dataset::{DenseRowMajor, Finite, TargetState, Targets};
use predict::metrics::{Accuracy, MSE, Metric};
use predict::selection::{
    Classes, CrossValidation, KFold, Split, Splitter, StratifiedKFold, TrainTest, TrainTestSplit,
};
use predict::{Dataset, Fit, Predict, PredictError, Result, Seed, ValidateConfig};

mod common;
use common::only_dense;

/// Every split partitions all rows, both sides ascending.
fn assert_partitions(splits: &[Split], n_samples: usize) {
    for split in splits {
        assert_eq!(split.n_samples, n_samples);
        assert!(split.train.is_sorted() && split.test.is_sorted());
        let mut all = [split.train.clone(), split.test.clone()].concat();
        all.sort_unstable();
        assert_eq!(all, (0..n_samples).collect::<Vec<_>>());
    }
}

/// The test sides of K-fold splits cover every row exactly once.
fn assert_each_row_tested_once(splits: &[Split], n_samples: usize) {
    let mut tested: Vec<usize> = splits.iter().flat_map(|s| s.test.clone()).collect();
    tested.sort_unstable();
    assert_eq!(tested, (0..n_samples).collect::<Vec<_>>());
}

fn is_invalid(result: Result<Vec<Split>>, expected: &str) -> bool {
    matches!(result, Err(PredictError::InvalidParameter { name, .. }) if name == expected)
}

// --- Split::select ------------------------------------------------------------

#[test]
fn select_gives_each_side_its_rows_targets_and_weights() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let (y, w) = (array![10.0, 11.0, 12.0, 13.0], array![1.0, 2.0, 3.0, 4.0]);
    let data = Dataset::new(&x)
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap();
    let split = Split {
        n_samples: 4,
        train: vec![0, 2],
        test: vec![3],
    };
    let TrainTest { train, test } = split.select(&data);
    assert_eq!(only_dense(&train), array![[0.0], [2.0]]);
    assert_eq!(train.y(), array![10.0, 12.0]);
    assert_eq!(train.sample_weights().unwrap(), array![1.0, 3.0]);
    assert_eq!(only_dense(&test), array![[3.0]]);
    assert_eq!(test.y(), array![13.0]);
}

#[test]
#[should_panic(expected = "this split was made for 3 rows, not 4")]
fn select_panics_on_data_of_another_size() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let split = Split {
        n_samples: 3,
        train: vec![0, 1],
        test: vec![2],
    };
    split.select(&Dataset::new(&x));
}

// --- TrainTestSplit -----------------------------------------------------------

#[test]
fn train_test_split_rounds_the_test_side_up() {
    let splits = TrainTestSplit::new().with_seed(1).splits(10, ()).unwrap();
    assert_eq!(splits.len(), 1);
    assert_partitions(&splits, 10);
    // ceil(0.25 * 10) = 3
    assert_eq!(splits[0].test.len(), 3);

    let split = TrainTestSplit::new()
        .with_test_size(0.07)
        .with_seed(1)
        .splits(100, ())
        .unwrap();
    // 0.07 * 100 is 7.000000000000001 in f64, as in scikit-learn.
    assert_eq!(split[0].test.len(), 8);
}

#[test]
fn train_test_split_repeats_with_a_seed_and_shuffles_otherwise() {
    let draw = |splitter: TrainTestSplit| splitter.splits(100, ()).unwrap().remove(0);
    let fixed = TrainTestSplit::new().with_seed(7);
    assert_eq!(draw(fixed.clone()), draw(fixed));
    assert_ne!(
        draw(TrainTestSplit::new().with_seed(1)),
        draw(TrainTestSplit::new().with_seed(2))
    );
    // Shuffled, the test side isn't simply the last rows.
    assert_ne!(
        draw(TrainTestSplit::new().with_seed(1)).test,
        (75..100).collect::<Vec<_>>()
    );
}

#[test]
fn train_test_split_without_shuffle_tests_on_the_last_rows_and_borrows() {
    let split = TrainTestSplit::new()
        .with_test_size(0.4)
        .without_shuffle()
        .splits(5, ())
        .unwrap()
        .remove(0);
    assert_eq!(split.train, vec![0, 1, 2]);
    assert_eq!(split.test, vec![3, 4]);

    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
    let data = Dataset::new(x.clone());
    let TrainTest { train, test } = split.select(&data);
    let base = only_dense(&data).as_ptr();
    // Each side is one range of rows, so both point into `data`.
    assert_eq!(only_dense(&train).as_ptr(), base);
    assert_eq!(only_dense(&test).as_ptr(), base.wrapping_add(3));
}

#[test]
fn train_test_split_checks_test_size() {
    for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let result = TrainTestSplit::new().with_test_size(bad).splits(10, ());
        assert!(is_invalid(result, "test_size"), "{bad}");
    }
    // ceil(0.5 * 1) = 1 leaves no row to train on, as does no row at all.
    assert!(is_invalid(
        TrainTestSplit::new().with_test_size(0.5).splits(1, ()),
        "test_size"
    ));
    assert!(is_invalid(TrainTestSplit::new().splits(0, ()), "test_size"));
}

// --- KFold --------------------------------------------------------------------

#[test]
fn kfold_cuts_consecutive_folds_the_first_ones_larger() {
    let splits = KFold::new().with_n_splits(3).splits(10, ()).unwrap();
    let tests: Vec<_> = splits.iter().map(|s| s.test.clone()).collect();
    assert_eq!(tests, vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
    assert_partitions(&splits, 10);
    assert_eq!(splits[1].train, vec![0, 1, 2, 3, 7, 8, 9]);
}

#[test]
fn kfold_shuffled_still_tests_each_row_once() {
    let kfold = KFold::new().with_shuffle(Seed::Fixed(3));
    let splits = kfold.splits(23, ()).unwrap();
    assert_eq!(splits.len(), 5);
    assert_partitions(&splits, 23);
    assert_each_row_tested_once(&splits, 23);
    assert_eq!(splits, kfold.splits(23, ()).unwrap());
    assert_ne!(splits, KFold::new().splits(23, ()).unwrap());
}

#[test]
fn kfold_checks_n_splits() {
    assert!(is_invalid(
        KFold::new().with_n_splits(1).splits(10, ()),
        "n_splits"
    ));
    assert!(is_invalid(
        KFold::new().with_n_splits(11).splits(10, ()),
        "n_splits"
    ));
    assert!(KFold::new().with_n_splits(10).splits(10, ()).is_ok());
}

// --- StratifiedKFold ----------------------------------------------------------

#[test]
fn stratified_kfold_deals_each_class_in_turn() {
    let classes = array![0_usize, 0, 0, 0, 1, 1, 1, 1];
    let splits = StratifiedKFold::new()
        .with_n_splits(3)
        .splits(8, Classes(classes.view()))
        .unwrap();
    let tests: Vec<_> = splits.iter().map(|s| s.test.clone()).collect();
    assert_eq!(tests, vec![vec![0, 3, 6], vec![1, 4, 7], vec![2, 5]]);
    assert_partitions(&splits, 8);
}

#[test]
fn stratified_kfold_keeps_class_proportions_when_shuffled() {
    // 30 rows of class 0 and 10 of class 2, interleaved.
    let classes: Array1<usize> = (0..40).map(|i| if i % 4 == 3 { 2 } else { 0 }).collect();
    let splits = StratifiedKFold::new()
        .with_shuffle(Seed::Fixed(5))
        .splits(40, Classes(classes.view()))
        .unwrap();
    assert_partitions(&splits, 40);
    assert_each_row_tested_once(&splits, 40);
    for split in &splits {
        let minority = split.test.iter().filter(|&&row| classes[row] == 2).count();
        assert_eq!((split.test.len(), minority), (8, 2));
    }
}

#[test]
fn stratified_kfold_checks_its_inputs_in_order() {
    let classes = array![0_usize, 0, 1, 1, 1];
    // Wrong length first, even with a bad n_splits.
    let result = StratifiedKFold::new()
        .with_n_splits(1)
        .splits(4, Classes(classes.view()));
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "classes",
            ..
        })
    ));
    let result = StratifiedKFold::new()
        .with_n_splits(6)
        .splits(5, Classes(classes.view()));
    assert!(is_invalid(result, "n_splits"));
    // Class 0 has 2 rows, too few for 3 folds.
    let result = StratifiedKFold::new()
        .with_n_splits(3)
        .splits(5, Classes(classes.view()));
    assert!(is_invalid(result, "classes"));
}

// --- CrossValidation ----------------------------------------------------------

/// A baseline regressor: predicts the training targets' weighted mean.
#[derive(Debug, Default)]
struct Mean {
    fail: bool,
}

struct FittedMean(f64);

impl ValidateConfig for Mean {
    fn validate_config(&self) -> Result<()> {
        if self.fail {
            return Err(PredictError::InvalidParameter {
                name: "fail",
                reason: "set for the test".into(),
            });
        }
        Ok(())
    }
}

impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for Mean {
    type Fitted = FittedMean;

    fn fit(
        &self,
        data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>,
    ) -> Result<FittedMean> {
        self.validate_config()?;
        let y = data.y();
        let mean = match data.sample_weights() {
            Some(w) => y.dot(&w) / w.sum(),
            None => y.mean().unwrap(),
        };
        Ok(FittedMean(mean))
    }
}

impl<'a, Y: TargetState> Predict<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>> for FittedMean {
    type Output = Array1<f64>;

    fn predict(&self, x: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(x.n_samples(), self.0))
    }
}

/// A baseline classifier: predicts the most common training class.
struct MostCommon;

struct FittedMostCommon(usize);

impl ValidateConfig for MostCommon {}

impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, usize>>> for MostCommon {
    type Fitted = FittedMostCommon;

    fn fit(
        &self,
        data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, usize>>,
    ) -> Result<FittedMostCommon> {
        let y = data.y();
        let count = |class: usize| y.iter().filter(|&&c| c == class).count();
        let most = (0..=*y.iter().max().unwrap())
            .max_by_key(|&c| count(c))
            .unwrap();
        Ok(FittedMostCommon(most))
    }
}

impl<'a, Y: TargetState> Predict<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>>
    for FittedMostCommon
{
    type Output = Array1<usize>;

    fn predict(&self, x: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<Array1<usize>> {
        Ok(Array1::from_elem(x.n_samples(), self.0))
    }
}

#[test]
fn cross_validation_scores_each_split_in_order() {
    let (x, y) = (
        array![[0.0], [0.0], [0.0], [0.0]],
        array![1.0, 2.0, 3.0, 4.0],
    );
    let data = Dataset::new(&x)
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap();
    let cv = CrossValidation::new(KFold::new().with_n_splits(2));
    let scores = cv.score(&Mean::default(), &data, (), &MSE).unwrap();
    // Fold 0 tests rows 0, 1 against the mean of 3 and 4; fold 1 the reverse.
    assert_eq!(scores, array![4.25, 4.25]);
}

#[test]
fn cross_validation_fits_and_scores_with_weights() {
    let (x, y) = (
        array![[0.0], [0.0], [0.0], [0.0]],
        array![1.0, 2.0, 3.0, 4.0],
    );
    let w = array![1.0, 1.0, 1.0, 3.0];
    let data = Dataset::new(&x)
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap();
    let cv = CrossValidation::new(KFold::new().with_n_splits(2));
    let scores = cv.score(&Mean::default(), &data, (), &MSE).unwrap();
    // Fold 0: fitted on rows 2, 3 weighing 1, 3, so the mean is 3.75; scored
    // on rows 0, 1: (2.75² + 1.75²) / 2.
    // Fold 1: fitted on rows 0, 1, mean 1.5; scored on rows 2, 3 weighing 1,
    // 3: (1.5² + 3 × 2.5²) / 4.
    assert_eq!(scores, array![5.3125, 5.25]);
}

#[test]
fn cross_validation_stratifies_classification_and_takes_dyn_metrics() {
    let x = Array1::<f64>::zeros(12).insert_axis(ndarray::Axis(1));
    let y = array![0_usize, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    let data = Dataset::new(&x)
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap();
    let cv = CrossValidation::new(StratifiedKFold::new().with_n_splits(3));
    let metric: &dyn Metric<usize, f64> = &Accuracy;
    let scores = cv
        .score(&MostCommon, &data, Classes(data.y()), metric)
        .unwrap();
    // Every test fold holds 3 of class 0 and 1 of class 1.
    assert_eq!(scores, array![0.75, 0.75, 0.75]);
}

#[test]
fn cross_validation_returns_the_first_error() {
    let (x, y) = (array![[0.0], [0.0]], array![1.0, 2.0]);
    let data = Dataset::new(&x)
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap();
    let splitter_error =
        CrossValidation::new(KFold::new()).score(&Mean::default(), &data, (), &MSE);
    assert!(matches!(
        splitter_error,
        Err(PredictError::InvalidParameter {
            name: "n_splits",
            ..
        })
    ));
    let fit_error = CrossValidation::new(KFold::new().with_n_splits(2)).score(
        &Mean { fail: true },
        &data,
        (),
        &MSE,
    );
    assert!(matches!(
        fit_error,
        Err(PredictError::InvalidParameter { name: "fail", .. })
    ));
}
