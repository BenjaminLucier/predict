use ndarray::{Array1, array};
use predict::{PredictError, metrics};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

fn example() -> (Array1<f64>, Array1<f64>) {
    (array![3.0, -0.5, 2.0, 7.0], array![2.5, 0.0, 2.0, 8.0])
}

fn is_undefined(result: predict::Result<f64>, expected_context: &str) -> bool {
    matches!(result, Err(PredictError::Undefined { context, .. }) if context == expected_context)
}

#[test]
fn r2_compares_against_predicting_the_mean() {
    let (y_true, y_pred) = example();
    // mean = 2.875; SS_tot = 29.1875; SS_res = 1.5. scikit-learn: 0.9486081370449679.
    let r2 = metrics::r2_score(y_true.view(), y_pred.view()).unwrap();
    assert_close(r2, 1.0 - 1.5 / 29.1875);
    assert_close(r2, 0.9486081370449679);
}

#[test]
fn r2_landmarks() {
    let y_true = array![1.0, 2.0, 3.0];
    let r2 = |y_pred: Array1<f64>| metrics::r2_score(y_true.view(), y_pred.view()).unwrap();
    assert_close(r2(y_true.clone()), 1.0);
    assert_close(r2(array![2.0, 2.0, 2.0]), 0.0);
    // SS_res = 4 + 0 + 4 = 8, SS_tot = 2
    assert_close(r2(array![3.0, 2.0, 1.0]), -3.0);
}

#[test]
fn r2_on_constant_targets_is_undefined_even_when_perfect() {
    // scikit-learn returns 1.0 for the perfect case and 0.0 for the other.
    let y_true = array![5.0, 5.0];
    for y_pred in [array![5.0, 5.0], array![4.0, 6.0]] {
        let result = metrics::r2_score(y_true.view(), y_pred.view());
        assert!(is_undefined(result, "r2_score"));
    }
}

#[test]
fn r2_detects_constants_the_mean_rounds_away_from() {
    // The sum 0.30000000000000004 / 3 is not 0.1, so SS_tot is not exactly zero.
    let y_true = array![0.1, 0.1, 0.1];
    assert_ne!(y_true.sum() / 3.0, 0.1);
    let result = metrics::r2_score(y_true.view(), y_true.view());
    assert!(is_undefined(result, "r2_score"));
}

#[test]
fn r2_on_a_single_row_is_undefined() {
    let y = array![1.0];
    assert!(is_undefined(
        metrics::r2_score(y.view(), y.view()),
        "r2_score"
    ));
}

#[test]
fn r2_is_undefined_when_the_total_sum_of_squares_overflows_or_underflows() {
    let y_pred = array![0.0, 0.0];
    let overflow = array![1e300, -1e300];
    let result = metrics::r2_score(overflow.view(), y_pred.view());
    assert!(is_undefined(result, "r2_score"));
    let underflow = array![1e-200, 2e-200];
    let result = metrics::r2_score(underflow.view(), underflow.view());
    assert!(is_undefined(result, "r2_score"));
}

#[test]
fn r2_of_hopeless_predictions_is_negative_infinity() {
    // SS_res overflows; SS_tot is 0.5.
    let r2 = metrics::r2_score(array![0.0, 1.0].view(), array![1e200, 0.0].view()).unwrap();
    assert_eq!(r2, f64::NEG_INFINITY);
    // Both sums are finite, but their ratio is not.
    let r2 = metrics::r2_score(array![0.0, 1e-100].view(), array![1e150, 0.0].view()).unwrap();
    assert_eq!(r2, f64::NEG_INFINITY);
}

#[test]
fn r2_works_on_f32() {
    let y_true = array![1.0_f32, 2.0, 3.0];
    let y_pred = array![2.0_f32, 2.0, 2.0];
    assert_close(
        metrics::r2_score(y_true.view(), y_pred.view()).unwrap(),
        0.0,
    );
}

#[test]
fn weighted_r2_matches_scikit_learn() {
    let (y_true, y_pred) = example();
    let weights = array![2.0, 1.0, 0.0, 3.0];
    let r2 = metrics::r2_score_weighted(y_true.view(), y_pred.view(), weights.view()).unwrap();
    assert_close(r2, 0.9222126188418324);
}

#[test]
fn weighted_r2_with_integer_weights_equals_repeated_rows() {
    let (y_true, y_pred) = example();
    let weights = array![2.0, 1.0, 0.0, 3.0];
    let repeated_true = array![3.0, 3.0, -0.5, 7.0, 7.0, 7.0];
    let repeated_pred = array![2.5, 2.5, 0.0, 8.0, 8.0, 8.0];
    assert_close(
        metrics::r2_score_weighted(y_true.view(), y_pred.view(), weights.view()).unwrap(),
        metrics::r2_score(repeated_true.view(), repeated_pred.view()).unwrap(),
    );
}

#[test]
fn weighted_r2_ignores_zero_weight_rows_when_testing_for_a_constant() {
    let y_true = array![1.0, 1.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0];
    let weights = array![1.0, 1.0, 0.0];
    let result = metrics::r2_score_weighted(y_true.view(), y_pred.view(), weights.view());
    assert!(is_undefined(result, "r2_score_weighted"));
}

#[test]
fn weighted_r2_checks_its_weights() {
    let (y_true, y_pred) = example();
    let zeros = Array1::<f64>::zeros(4);
    let result = metrics::r2_score_weighted(y_true.view(), y_pred.view(), zeros.view());
    assert!(is_undefined(result, "r2_score_weighted"));
    let result = metrics::r2_score_weighted(y_true.view(), y_pred.view(), array![1.0].view());
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "sample_weights",
            ..
        })
    ));
}
