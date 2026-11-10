use ndarray::{Array, Array1, Array2, Ix2, ShapeBuilder, array, s};
use predict::dataset::{DenseRowMajor, Finite, NoInfinity, Target, TargetDim, Targets, Unchecked};
use predict::{Dataset, Float, PredictError};

// --- Borrowed or owned, never copied ---------------------------------------

#[test]
fn targets_are_attached_without_copying() {
    let x = array![[1.0], [2.0]];
    let y = array![3.0, 4.0];

    // Borrowed by reference or by view: the dataset reads `y` in place.
    assert_eq!(
        Dataset::new(&x).with_y(&y).unwrap().y().as_ptr(),
        y.as_ptr()
    );
    assert_eq!(
        Dataset::new(&x).with_y(y.view()).unwrap().y().as_ptr(),
        y.as_ptr()
    );

    // Moved in: the dataset owns the same buffer.
    let ptr = y.as_ptr();
    assert_eq!(Dataset::new(&x).with_y(y).unwrap().y().as_ptr(), ptr);

    // Several outputs keep their rows × outputs shape.
    let y = array![[20.0, 0.5], [21.0, 0.6]];
    let data: Dataset<'_, _, _, _, Targets<'_, f64, Ix2>> = Dataset::new(&x).with_y(&y).unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());
    assert_eq!(data.y(), y);
}

#[test]
fn a_column_of_a_table_can_be_the_targets() {
    // Features and targets often come from one table. The target column is a
    // strided view of it: attached in place, with errors reported in its own rows.
    let table = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let data = Dataset::new(table.slice(s![.., ..1]))
        .with_y(table.column(1))
        .unwrap();
    assert_eq!(data.y().as_ptr(), table.column(1).as_ptr());
    assert_eq!(data.y(), array![10.0, 20.0, 30.0]);

    let table = array![[1.0, 10.0], [2.0, f64::NAN], [3.0, 30.0]];
    let err = Dataset::new(table.slice(s![.., ..1]))
        .with_y(table.column(1))
        .unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite {
            context: "y",
            row: 1,
            col: 0,
            ..
        }
    ));
}

#[test]
fn owned_features_take_borrowed_targets() {
    // Owned features borrow nothing; attaching a borrowed `y` gives the dataset
    // the lifetime of that borrow, with no copy of either array.
    let y = array![0.0, 1.0];
    let data = Dataset::new(array![[1.0], [2.0]]).with_y(&y).unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());
}

#[test]
fn a_static_dataset_takes_borrowed_targets() {
    // A `'static` dataset shortens to the borrow of `y`, as a `&'static` reference
    // would, so it can take targets that live for less time.
    let features: Dataset<'static> = Dataset::new(array![[1.0], [2.0]]);
    let y = array![0.0, 1.0];
    let data = features.with_y(&y).unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());
}

/// `Dataset` and `Targets` shorten like references: a longer-lived one is accepted
/// where a shorter-lived one is expected. Checked at compile time: if a change to
/// ndarray made them invariant again, these functions would stop compiling.
#[test]
fn datasets_shorten_like_references() {
    fn features<'a>(data: Dataset<'static>) -> Dataset<'a> {
        data
    }
    fn with_targets<'a>(
        data: Dataset<'static, f64, DenseRowMajor<'static>, Unchecked, Targets<'static, f64>>,
    ) -> Dataset<'a, f64, DenseRowMajor<'a>, Unchecked, Targets<'a, f64>> {
        data
    }
    let data = with_targets(Dataset::new(array![[1.0]]).with_y(array![2.0]).unwrap());
    assert_eq!(data.y(), array![2.0]);
    assert_eq!(features(Dataset::new(array![[1.0]])).n_samples(), 1);
}

#[test]
fn owned_features_and_targets_make_a_static_dataset() {
    // Nothing is borrowed, so the dataset can leave the function that made it,
    // as a synthetic data generator needs.
    fn make() -> Dataset<'static, f64, DenseRowMajor<'static>, Unchecked, Targets<'static, f64>> {
        Dataset::new(array![[1.0], [2.0]])
            .with_y(array![3.0, 4.0])
            .unwrap()
    }
    assert_eq!(make().y(), array![3.0, 4.0]);
}

// --- Shape: rows first, then outputs ----------------------------------------

#[derive(Debug)]
enum Expect {
    Accepted,
    Mismatch { expected: usize, actual: usize },
    NoOutputs,
}

/// Attach `y` to `n_samples` rows of features, keeping only the outcome.
fn attach<D: TargetDim>(n_samples: usize, y: Array<f64, D>) -> Result<(), PredictError> {
    let x = Array2::<f64>::zeros((n_samples, 1));
    Dataset::new(&x).with_y(&y).map(|_| ())
}

fn assert_outcome(case: &str, result: Result<(), PredictError>, expect: Expect) {
    let as_expected = match (&result, &expect) {
        (Ok(()), Expect::Accepted) => true,
        (
            Err(PredictError::ShapeMismatch {
                context: "y",
                expected,
                actual,
            }),
            Expect::Mismatch {
                expected: e,
                actual: a,
            },
        ) => (expected, actual) == (e, a),
        (Err(PredictError::EmptyInput { context: "y" }), Expect::NoOutputs) => true,
        _ => false,
    };
    assert!(as_expected, "{case}: expected {expect:?}, got {result:?}");
}

#[test]
fn shapes_are_checked_rows_first_then_outputs() {
    use Expect::*;

    // (case, samples, y, expected outcome)
    let one_target_per_row = [
        ("one per sample", 3, array![1.0, 2.0, 3.0], Accepted),
        ("no samples, no targets", 0, Array1::zeros(0), Accepted),
        (
            "too few",
            3,
            array![1.0, 2.0],
            Mismatch {
                expected: 3,
                actual: 2,
            },
        ),
        (
            "too many",
            3,
            array![1.0, 2.0, 3.0, 4.0],
            Mismatch {
                expected: 3,
                actual: 4,
            },
        ),
        (
            "the length is checked before the values",
            3,
            array![f64::NAN, 1.0],
            Mismatch {
                expected: 3,
                actual: 2,
            },
        ),
    ];
    for (case, n_samples, y, expect) in one_target_per_row {
        assert_outcome(case, attach(n_samples, y), expect);
    }

    let several_outputs = [
        ("rows × outputs", 3, Array2::zeros((3, 2)), Accepted),
        ("a single output column", 3, Array2::zeros((3, 1)), Accepted),
        (
            "too few rows",
            3,
            Array2::zeros((2, 2)),
            Mismatch {
                expected: 3,
                actual: 2,
            },
        ),
        ("no outputs", 3, Array2::zeros((3, 0)), NoOutputs),
        (
            "rows are checked before outputs",
            3,
            Array2::zeros((2, 0)),
            Mismatch {
                expected: 3,
                actual: 2,
            },
        ),
    ];
    for (case, n_samples, y, expect) in several_outputs {
        assert_outcome(case, attach(n_samples, y), expect);
    }
}

// --- Values: what each element type allows ----------------------------------

/// Every kind of special value as a float target, 1-D and 2-D in both layouts.
#[test]
fn each_special_value_as_a_float_target() {
    float_targets::<f64>();
    float_targets::<f32>();
}

fn float_targets<F: Float + Target>() {
    // (name, value, accepted)
    let cases = [
        ("NaN", F::nan(), false),
        ("+infinity", F::infinity(), false),
        ("-infinity", F::neg_infinity(), false),
        ("largest", F::max_value(), true),
        ("most negative", F::min_value(), true),
        ("smallest normal", F::min_positive_value(), true),
        (
            "subnormal",
            F::min_positive_value() / (F::one() + F::one()),
            true,
        ),
        ("negative zero", F::neg_zero(), true),
    ];
    let x = Array2::<f64>::zeros((2, 1));

    for (name, value, accepted) in cases {
        // 1-D: the value in row 1.
        let mut y = Array1::<F>::zeros(2);
        y[1] = value;
        let result = Dataset::new(&x).with_y(&y).map(|_| ());
        assert_value(&format!("1-D, {name}"), result, accepted, (1, 0), value);

        // 2-D, row-major and column-major: the value in row 1, output 2.
        for y in [Array2::<F>::zeros((2, 3)), Array2::<F>::zeros((2, 3).f())] {
            let mut y = y;
            y[[1, 2]] = value;
            let result = Dataset::new(&x).with_y(&y).map(|_| ());
            assert_value(&format!("2-D, {name}"), result, accepted, (1, 2), value);
        }
    }
}

fn assert_value<F: Float>(
    case: &str,
    result: Result<(), PredictError>,
    accepted: bool,
    at: (usize, usize),
    value: F,
) {
    match result {
        Ok(()) => assert!(accepted, "{case}: accepted, should be rejected"),
        Err(PredictError::NonFinite {
            context: "y",
            row,
            col,
            value: reported,
        }) => {
            assert!(!accepted, "{case}: rejected, should be accepted");
            assert_eq!((row, col), at, "{case}: wrong position");
            let value: f64 = value.into();
            assert!(
                reported == value || (reported.is_nan() && value.is_nan()),
                "{case}: reported {reported}, expected {value}"
            );
        }
        Err(other) => panic!("{case}: unexpected error {other:?}"),
    }
}

#[test]
fn every_class_index_is_accepted() {
    // A `usize` can't be negative or NaN, and `with_y` can't know how many classes
    // there are, so every value passes; classifiers check the range in `fit`.
    let x = array![[1.0], [2.0]];
    let classes = array![0, usize::MAX];
    assert_eq!(Dataset::new(&x).with_y(&classes).unwrap().y(), classes);

    // Multi-label: two samples, three yes / no labels each.
    let labels = array![[1_usize, 0, 1], [0, 0, 1]];
    assert_eq!(Dataset::new(&x).with_y(&labels).unwrap().y(), labels);
}

// --- Targets and the feature state are independent ---------------------------

#[test]
fn targets_and_the_feature_state_are_independent() {
    // NaN is allowed in `x` (`NoInfinity`), never in `y`.
    let x = array![[1.0, f64::NAN], [2.0, 3.0]];
    let (y, bad_y) = (array![1.0, 2.0], array![1.0, f64::NAN]);

    // Validate, then attach: the targets are still checked.
    let features = Dataset::new(&x).validate_no_infinity().unwrap();
    let err = features.clone().with_y(&bad_y).unwrap_err();
    assert!(matches!(err, PredictError::NonFinite { context: "y", .. }));
    let data: Dataset<'_, _, _, NoInfinity, Targets<'_, f64>> = features.with_y(&y).unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());

    // Attach, then validate: validation checks only `x` and keeps `y` as it was.
    let data: Dataset<'_, _, _, NoInfinity, Targets<'_, f64>> = Dataset::new(&x)
        .with_y(&y)
        .unwrap()
        .validate_no_infinity()
        .unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());

    // A feature error is still reported against `x`, not the targets.
    let err = data.validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite {
            context: "validate_finite",
            row: 0,
            col: 1,
            ..
        }
    ));

    // All the way to `Finite`, with class targets.
    let x = array![[1.0], [2.0]];
    let classes = array![0_usize, 1];
    let data: Dataset<'_, _, _, Finite, Targets<'_, usize>> = Dataset::new(&x)
        .with_y(&classes)
        .unwrap()
        .validate_finite()
        .unwrap();
    assert_eq!(data.y().as_ptr(), classes.as_ptr());
}
