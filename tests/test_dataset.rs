use ndarray::{Array2, ShapeBuilder, array};
use predict::{Dataset, Float, PredictError};

#[test]
fn new_borrows_without_copying() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let data = Dataset::new(x.view());

    // Same memory: the dataset reads the caller's array in place.
    assert_eq!(data.view().as_ptr(), x.as_ptr());
    assert_eq!((data.n_samples(), data.n_features()), (3, 2));
}

#[test]
fn from_owned_outlives_the_original_binding() {
    let data = {
        let x = Array2::<f64>::zeros((4, 3));
        Dataset::from_owned(x)
    };
    assert_eq!((data.n_samples(), data.n_features()), (4, 3));
}

#[test]
fn the_float_type_is_inferred_from_the_array() {
    let x = array![[1.0_f32, 2.0]];
    let data = Dataset::new(x.view());
    let view: ndarray::ArrayView2<'_, f32> = data.view();
    assert_eq!(view[[0, 1]], 2.0);
}

#[test]
fn validate_finite_reports_the_first_bad_value() {
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::INFINITY, 6.0]];
    let err = Dataset::new(x.view()).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite {
            context: "validate_finite",
            row: 1,
            col: 1,
            ..
        }
    ));
}

#[test]
fn a_single_bad_value_is_located_in_either_layout() {
    for x in [
        Array2::<f64>::zeros((3, 2)),
        Array2::<f64>::zeros((3, 2).f()),
    ] {
        let mut x = x;
        x[[2, 1]] = f64::NAN;
        let err = Dataset::new(x.view()).validate_finite().unwrap_err();
        assert!(matches!(
            err,
            PredictError::NonFinite { row: 2, col: 1, .. }
        ));
    }
}

#[test]
fn with_several_bad_values_the_first_in_memory_is_reported() {
    // A 3x2 column-major array stores column 0 first, so (2, 0) comes before
    // (0, 1) in memory, although (0, 1) comes first reading row by row.
    let mut x = Array2::<f64>::zeros((3, 2).f());
    x[[0, 1]] = f64::NAN;
    x[[2, 0]] = f64::NAN;
    let err = Dataset::new(x.view()).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 2, col: 0, .. }
    ));
}

#[test]
fn validation_moves_the_data_without_copying() {
    let x = array![[1.0, 2.0]];
    let finite = Dataset::new(x.view()).validate_finite().unwrap();
    assert_eq!(finite.view().as_ptr(), x.as_ptr());
}

#[test]
fn f32_values_are_reported_as_f64() {
    let x = array![[f32::INFINITY]];
    let err = Dataset::new(x.view()).validate_finite().unwrap_err();
    assert!(matches!(err, PredictError::NonFinite { value, .. } if value == f64::INFINITY));
}

/// Every kind of special value against each check, on all three paths: `Unchecked
/// -> NoInfinity`, `Unchecked -> Finite`, and `NoInfinity -> Finite`.
#[test]
fn each_special_value_against_each_check() {
    special_values::<f64>();
    special_values::<f32>();
}

fn special_values<F: Float>() {
    // (name, value, passes `validate_no_infinity`, passes `validate_finite`)
    let cases = [
        ("NaN", F::nan(), true, false),
        ("+infinity", F::infinity(), false, false),
        ("-infinity", F::neg_infinity(), false, false),
        ("largest", F::max_value(), true, true),
        ("most negative", F::min_value(), true, true),
        ("smallest normal", F::min_positive_value(), true, true),
        (
            "subnormal",
            F::min_positive_value() / (F::one() + F::one()),
            true,
            true,
        ),
        ("negative zero", F::neg_zero(), true, true),
    ];

    for (name, value, no_infinity_ok, finite_ok) in cases {
        let x = array![[F::one(), value]];

        let no_infinity = Dataset::new(x.view()).validate_no_infinity();
        assert_eq!(
            no_infinity.is_ok(),
            no_infinity_ok,
            "validate_no_infinity, {name}"
        );
        if let Err(err) = &no_infinity {
            assert!(matches!(
                err,
                PredictError::NonFinite {
                    context: "validate_no_infinity",
                    row: 0,
                    col: 1,
                    ..
                }
            ));
        }

        let finite = Dataset::new(x.view()).validate_finite();
        assert_eq!(
            finite.is_ok(),
            finite_ok,
            "validate_finite from Unchecked, {name}"
        );
        if let Err(err) = &finite {
            assert!(matches!(
                err,
                PredictError::NonFinite {
                    context: "validate_finite",
                    row: 0,
                    col: 1,
                    ..
                }
            ));
        }

        // NoInfinity -> Finite agrees with the direct check.
        if let Ok(data) = no_infinity {
            assert_eq!(
                data.validate_finite().is_ok(),
                finite_ok,
                "validate_finite from NoInfinity, {name}"
            );
        }
    }
}
