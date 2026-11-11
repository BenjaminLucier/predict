use ndarray::{Array1, array};
use predict::{PredictError, metrics};

fn example() -> (Array1<f64>, Array1<f64>) {
    (array![3.0, -0.5, 2.0, 7.0], array![2.5, 0.0, 2.0, 8.0])
}

#[test]
fn mse_is_the_mean_of_squared_residuals() {
    let (y_true, y_pred) = example();
    // (0.25 + 0.25 + 0 + 1) / 4
    assert_eq!(metrics::mse(y_true.view(), y_pred.view()).unwrap(), 0.375);
}

#[test]
fn rmse_is_the_root_of_mse() {
    let (y_true, y_pred) = example();
    let rmse = metrics::rmse(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(rmse, 0.375_f64.sqrt());
}

#[test]
fn mae_is_the_mean_of_absolute_residuals() {
    let (y_true, y_pred) = example();
    // (0.5 + 0.5 + 0 + 1) / 4
    assert_eq!(metrics::mae(y_true.view(), y_pred.view()).unwrap(), 0.5);
}

#[test]
fn f32_inputs_give_f64_results() {
    let y_true = array![1.0_f32, 2.0];
    let y_pred = array![2.0_f32, 2.0];
    let mse: f64 = metrics::mse(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(mse, 0.5);
}

#[test]
fn f32_inputs_too_large_to_square_in_f32_do_not_overflow() {
    // 1e30² = 1e60, far beyond f32's ~3.4e38 but well inside f64.
    let y_true = array![1e30_f32];
    let y_pred = array![-1e30_f32];
    let mse = metrics::mse(y_true.view(), y_pred.view()).unwrap();
    let expected = (2.0 * f64::from(1e30_f32)).powi(2);
    assert_eq!(mse, expected);
}

#[test]
fn f64_overflow_is_infinite_not_nan() {
    let mse = metrics::mse(array![1e300].view(), array![-1e300].view()).unwrap();
    assert_eq!(mse, f64::INFINITY);
}

#[test]
fn rejects_different_lengths() {
    let result = metrics::mse(array![1.0, 2.0].view(), array![1.0].view());
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "y_pred",
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn rejects_empty_input() {
    let empty = Array1::<f64>::zeros(0);
    assert!(matches!(
        metrics::mae(empty.view(), empty.view()),
        Err(PredictError::EmptyInput { context: "y_true" })
    ));
}

#[test]
fn rejects_non_finite_values_at_their_row() {
    let result = metrics::mse(array![1.0, 2.0].view(), array![1.0, f64::INFINITY].view());
    assert!(matches!(
        result,
        Err(PredictError::NonFinite {
            context: "y_pred",
            row: 1,
            col: 0,
            ..
        })
    ));
    let result = metrics::mae(array![f64::NAN, 2.0].view(), array![1.0, 2.0].view());
    assert!(matches!(
        result,
        Err(PredictError::NonFinite {
            context: "y_true",
            row: 0,
            ..
        })
    ));
}

#[test]
fn integer_weights_equal_repeated_rows() {
    let (y_true, y_pred) = example();
    let weights = array![2.0, 1.0, 0.0, 3.0];
    // Row 0 twice, row 1 once, row 2 dropped, row 3 three times.
    let repeated_true = array![3.0, 3.0, -0.5, 7.0, 7.0, 7.0];
    let repeated_pred = array![2.5, 2.5, 0.0, 8.0, 8.0, 8.0];
    let (t, p, w) = (y_true.view(), y_pred.view(), weights.view());
    let (rt, rp) = (repeated_true.view(), repeated_pred.view());
    assert_eq!(
        metrics::mse_weighted(t, p, w).unwrap(),
        metrics::mse(rt, rp).unwrap()
    );
    assert_eq!(
        metrics::rmse_weighted(t, p, w).unwrap(),
        metrics::rmse(rt, rp).unwrap()
    );
    assert_eq!(
        metrics::mae_weighted(t, p, w).unwrap(),
        metrics::mae(rt, rp).unwrap()
    );
}

#[test]
fn uniform_weights_equal_the_unweighted_metric() {
    let (y_true, y_pred) = example();
    let weights = Array1::from_elem(4, 0.5);
    assert_eq!(
        metrics::mse_weighted(y_true.view(), y_pred.view(), weights.view()).unwrap(),
        metrics::mse(y_true.view(), y_pred.view()).unwrap()
    );
}

#[test]
fn weights_are_checked_like_dataset_weights() {
    let (y_true, y_pred) = example();
    let (t, p) = (y_true.view(), y_pred.view());
    assert!(matches!(
        metrics::mse_weighted(t, p, array![1.0, 1.0].view()),
        Err(PredictError::ShapeMismatch {
            context: "sample_weights",
            expected: 4,
            actual: 2
        })
    ));
    assert!(matches!(
        metrics::mae_weighted(t, p, array![1.0, -1.0, 1.0, 1.0].view()),
        Err(PredictError::InvalidValue {
            context: "sample_weights",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        metrics::mse_weighted(t, p, array![1.0, 1.0, f64::NAN, 1.0].view()),
        Err(PredictError::NonFinite {
            context: "sample_weights",
            row: 2,
            ..
        })
    ));
}

#[test]
fn all_zero_weights_are_undefined() {
    let (y_true, y_pred) = example();
    let zeros = Array1::<f64>::zeros(4);
    let (t, p, w) = (y_true.view(), y_pred.view(), zeros.view());
    assert!(matches!(
        metrics::mae_weighted(t, p, w),
        Err(PredictError::Undefined {
            context: "mae_weighted",
            ..
        })
    ));
    assert!(matches!(
        metrics::rmse_weighted(t, p, w),
        Err(PredictError::Undefined {
            context: "rmse_weighted",
            ..
        })
    ));
}
