use ndarray::{Array1, Array2, array, s};
use predict::dataset::{Finite, NoInfinity, Targets, Unchecked};
use predict::{Dataset, Float, PredictError};

// --- Uniform by default -----------------------------------------------------

#[test]
fn every_row_weighs_one_until_weights_are_set() {
    let (x, y) = (array![[1.0], [2.0]], array![3.0, 4.0]);

    // `None` all the way through: new, with targets, after validation.
    let data = Dataset::new(&x);
    assert!(data.sample_weights().is_none());
    let data = data.with_y(&y).unwrap();
    assert!(data.sample_weights().is_none());
    assert!(data.validate_finite().unwrap().sample_weights().is_none());
}

// --- Borrowed or owned, never copied ----------------------------------------

#[test]
fn weights_are_attached_without_copying() {
    let x = array![[1.0], [2.0]];
    let w = array![0.5, 2.0];

    // Borrowed by reference or by view: the dataset reads `w` in place.
    let data = Dataset::new(&x).with_sample_weights(&w).unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());
    let data = Dataset::new(&x).with_sample_weights(w.view()).unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());

    // Moved in: the dataset owns the same buffer.
    let ptr = w.as_ptr();
    let data = Dataset::new(&x).with_sample_weights(w).unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), ptr);
}

#[test]
fn a_column_of_a_table_can_be_the_weights() {
    // Features, targets and weights often come from one table: each is a view of
    // it, attached in place.
    let table = array![[1.0, 10.0, 0.5], [2.0, 20.0, 1.0], [3.0, 30.0, 2.0]];
    let data = Dataset::new(table.slice(s![.., ..1]))
        .with_y(table.column(1))
        .unwrap()
        .with_sample_weights(table.column(2))
        .unwrap();
    assert_eq!(
        data.sample_weights().unwrap().as_ptr(),
        table.column(2).as_ptr()
    );
    assert_eq!(data.sample_weights().unwrap(), array![0.5, 1.0, 2.0]);
}

#[test]
fn owned_data_with_owned_weights_makes_a_static_dataset() {
    // Nothing is borrowed, so the dataset can leave the function that made it.
    fn make() -> Dataset<'static, f64, Unchecked, Targets<'static, f64>> {
        Dataset::new(array![[1.0], [2.0]])
            .with_y(array![3.0, 4.0])
            .unwrap()
            .with_sample_weights(array![1.0, 2.0])
            .unwrap()
    }
    assert_eq!(make().sample_weights().unwrap(), array![1.0, 2.0]);
}

#[test]
fn a_static_dataset_takes_borrowed_weights() {
    // As with targets, a `'static` dataset shortens to the borrow of the weights.
    let data: Dataset<'static> = Dataset::new(array![[1.0], [2.0]]);
    let w = array![1.0, 2.0];
    let data = data.with_sample_weights(&w).unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());
}

// --- Length ------------------------------------------------------------------

#[test]
fn there_is_one_weight_per_sample() {
    // (case, samples, weights, Some((expected, actual)) for a mismatch)
    let cases = [
        ("one per sample", 3, array![1.0, 1.0, 1.0], None),
        ("no samples, no weights", 0, Array1::zeros(0), None),
        ("too few", 3, array![1.0, 1.0], Some((3, 2))),
        ("too many", 3, array![1.0, 1.0, 1.0, 1.0], Some((3, 4))),
        (
            "the length is checked before the values",
            3,
            array![-1.0, f64::NAN],
            Some((3, 2)),
        ),
    ];
    for (case, n_samples, w, mismatch) in cases {
        let x = Array2::<f64>::zeros((n_samples, 1));
        let result = Dataset::new(&x).with_sample_weights(&w).map(|_| ());
        match (result, mismatch) {
            (Ok(()), None) => {}
            (
                Err(PredictError::ShapeMismatch {
                    context: "sample_weights",
                    expected,
                    actual,
                }),
                Some(sizes),
            ) => assert_eq!((expected, actual), sizes, "{case}"),
            (result, mismatch) => panic!("{case}: expected {mismatch:?}, got {result:?}"),
        }
    }
}

// --- Values -------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Accepted,
    NonFinite,
    Negative,
}

/// Every kind of special value as a weight, in row 1 of three.
#[test]
fn each_special_value_as_a_weight() {
    weight_values::<f64>();
    weight_values::<f32>();
}

fn weight_values<F: Float>() {
    use Outcome::*;
    let cases = [
        ("NaN", F::nan(), NonFinite),
        ("+infinity", F::infinity(), NonFinite),
        ("-infinity", F::neg_infinity(), NonFinite),
        ("-1", -F::one(), Negative),
        ("most negative", F::min_value(), Negative),
        ("smallest negative", -F::min_positive_value(), Negative),
        ("zero", F::zero(), Accepted),
        ("negative zero", F::neg_zero(), Accepted),
        (
            "subnormal",
            F::min_positive_value() / (F::one() + F::one()),
            Accepted,
        ),
        ("largest", F::max_value(), Accepted),
    ];
    let x = Array2::<F>::zeros((3, 1));

    for (name, value, expected) in cases {
        let mut w = Array1::<F>::ones(3);
        w[1] = value;
        let result = Dataset::new(&x).with_sample_weights(&w).map(|_| ());
        let value: f64 = value.into();
        let same = |reported: f64| reported == value || (reported.is_nan() && value.is_nan());
        let outcome = match result {
            Ok(()) => Accepted,
            Err(PredictError::NonFinite {
                context: "sample_weights",
                row: 1,
                col: 0,
                value: reported,
            }) if same(reported) => NonFinite,
            Err(PredictError::InvalidValue {
                context: "sample_weights",
                row: 1,
                col: 0,
                value: reported,
                ref reason,
            }) if same(reported) && reason == "must be non-negative" => Negative,
            Err(other) => panic!("{name}: unexpected error {other:?}"),
        };
        assert_eq!(outcome, expected, "{name}");
    }
}

#[test]
fn the_first_bad_row_is_reported_whatever_the_problem() {
    // A negative weight before a NaN: the negative one is reported.
    let x = Array2::<f64>::zeros((3, 1));
    let w = array![1.0, -2.0, f64::NAN];
    let err = Dataset::new(&x).with_sample_weights(&w).unwrap_err();
    assert!(matches!(err, PredictError::InvalidValue { row: 1, .. }));
}

#[test]
fn all_zero_weights_are_left_for_fit_to_judge() {
    // A positive total is checked in `fit`, which must check it per fold anyway.
    let x = array![[1.0], [2.0]];
    let w = array![0.0, 0.0];
    assert!(Dataset::new(&x).with_sample_weights(&w).is_ok());
}

// --- Replacing, and independence from targets and the feature state ------------

#[test]
fn setting_weights_again_replaces_them() {
    let x = array![[1.0], [2.0]];
    let (first, second) = (array![1.0, 2.0], array![3.0, 4.0]);
    let data = Dataset::new(&x)
        .with_sample_weights(&first)
        .unwrap()
        .with_sample_weights(&second)
        .unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), second.as_ptr());
}

#[test]
fn weights_targets_and_the_feature_state_are_independent() {
    // NaN in `x` is allowed by `NoInfinity`; the weights don't care either way.
    let x = array![[1.0, f64::NAN], [2.0, 3.0]];
    let (y, w) = (array![0_usize, 1], array![1.0, 2.0]);

    // Weights before targets, then validation: all kept as they were.
    let data: Dataset<'_, _, NoInfinity, Targets<'_, usize>> = Dataset::new(&x)
        .with_sample_weights(&w)
        .unwrap()
        .with_y(&y)
        .unwrap()
        .validate_no_infinity()
        .unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());

    // Validation, then targets, then weights: the same.
    let data = Dataset::new(&x)
        .validate_no_infinity()
        .unwrap()
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap();
    assert_eq!(data.y().as_ptr(), y.as_ptr());
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());

    // A feature error is still reported against `x`.
    let err = data.validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite {
            context: "validate_finite",
            ..
        }
    ));

    // All the way to `Finite`, weights kept.
    let x = array![[1.0], [2.0]];
    let data: Dataset<'_, _, Finite> = Dataset::new(&x)
        .with_sample_weights(&w)
        .unwrap()
        .validate_finite()
        .unwrap();
    assert_eq!(data.sample_weights().unwrap().as_ptr(), w.as_ptr());
}
