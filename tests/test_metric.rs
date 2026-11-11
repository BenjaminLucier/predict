use ndarray::{ArrayView1, array};
use predict::PredictError;
use predict::metrics::{
    self, Accuracy, Average, Direction, F1, FBeta, MAE, MSE, Metric, Precision, R2, RMSE, Recall,
};

#[test]
fn regression_metrics_call_their_functions() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    let weights = array![2.0, 1.0, 0.0, 3.0];
    let (t, p, w) = (y_true.view(), y_pred.view(), weights.view());
    let unweighted: Option<ArrayView1<'_, f64>> = None;

    assert_eq!(
        MSE.compute(t, p, unweighted).unwrap(),
        metrics::mse(t, p).unwrap()
    );
    assert_eq!(
        RMSE.compute(t, p, unweighted).unwrap(),
        metrics::rmse(t, p).unwrap()
    );
    assert_eq!(
        MAE.compute(t, p, unweighted).unwrap(),
        metrics::mae(t, p).unwrap()
    );
    assert_eq!(
        R2.compute(t, p, unweighted).unwrap(),
        metrics::r2_score(t, p).unwrap()
    );

    assert_eq!(
        MSE.compute(t, p, Some(w)).unwrap(),
        metrics::mse_weighted(t, p, w).unwrap()
    );
    assert_eq!(
        R2.compute(t, p, Some(w)).unwrap(),
        metrics::r2_score_weighted(t, p, w).unwrap()
    );
}

#[test]
fn classification_metrics_call_their_functions() {
    let y_true = array![1, 1, 1, 0, 0];
    let y_pred = array![1, 1, 0, 1, 0];
    let weights = array![1.0_f32, 2.0, 1.0, 1.0, 3.0];
    let (t, p, w) = (y_true.view(), y_pred.view(), weights.view());
    let average = Average::Macro;

    assert_eq!(
        Accuracy.compute(t, p, Some(w)).unwrap(),
        metrics::accuracy_weighted(t, p, w).unwrap()
    );
    assert_eq!(
        Precision { average }.compute(t, p, Some(w)).unwrap(),
        metrics::precision_weighted(t, p, average, w).unwrap()
    );
    assert_eq!(
        Recall { average }.compute(t, p, Some(w)).unwrap(),
        metrics::recall_weighted(t, p, average, w).unwrap()
    );
    assert_eq!(
        F1 { average }.compute(t, p, Some(w)).unwrap(),
        metrics::f1_score_weighted(t, p, average, w).unwrap()
    );
    let fbeta = FBeta { beta: 2.0, average };
    assert_eq!(
        fbeta.compute(t, p, Some(w)).unwrap(),
        metrics::fbeta_score_weighted(t, p, 2.0, average, w).unwrap()
    );
}

#[test]
fn targets_and_weights_may_have_different_float_types() {
    let y = array![1.0_f32, 2.0, 3.0];
    let weights = array![1.0_f64, 1.0, 2.0];
    let mse = MSE
        .compute(y.view(), y.view(), Some(weights.view()))
        .unwrap();
    assert_eq!(mse, 0.0);
}

#[test]
fn directions_point_the_right_way() {
    fn direction<T>(metric: &dyn Metric<T, f64>) -> Direction {
        metric.direction()
    }
    for lower in [
        direction::<f64>(&MSE),
        direction::<f64>(&RMSE),
        direction::<f64>(&MAE),
    ] {
        assert_eq!(lower, Direction::LowerIsBetter);
    }
    assert_eq!(direction::<f64>(&R2), Direction::HigherIsBetter);
    let average = Average::Macro;
    let higher = [
        direction::<usize>(&Accuracy),
        direction(&Precision { average }),
        direction(&Recall { average }),
        direction(&F1 { average }),
        direction(&FBeta { beta: 1.0, average }),
    ];
    assert!(higher.iter().all(|&d| d == Direction::HigherIsBetter));
}

#[test]
fn metrics_of_different_types_fit_in_one_list() {
    let metrics: Vec<Box<dyn Metric<f64, f64>>> = vec![Box::new(MSE), Box::new(R2)];
    let y = array![1.0, 2.0, 3.0];
    let scores: Vec<f64> = metrics
        .iter()
        .map(|m| m.compute(y.view(), y.view(), None).unwrap())
        .collect();
    assert_eq!(scores, [0.0, 1.0]);
}

#[test]
fn errors_pass_through_and_beta_is_checked_when_computed() {
    let y = array![0, 1];
    let bad_beta = FBeta {
        beta: -1.0,
        average: Average::Macro,
    };
    let result = bad_beta.compute(y.view(), y.view(), None::<ArrayView1<'_, f64>>);
    assert!(matches!(
        result,
        Err(PredictError::InvalidParameter { name: "beta", .. })
    ));
}

/// A metric of the caller's own: the trait is open.
struct MaxError;

impl Metric<f64, f64> for MaxError {
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }

    fn compute(
        &self,
        y_true: ArrayView1<'_, f64>,
        y_pred: ArrayView1<'_, f64>,
        _sample_weights: Option<ArrayView1<'_, f64>>,
    ) -> predict::Result<f64> {
        let worst = y_true
            .iter()
            .zip(&y_pred)
            .map(|(t, p)| (t - p).abs())
            .fold(0.0, f64::max);
        Ok(worst)
    }
}

#[test]
fn callers_can_implement_their_own_metric() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.5, 2.0, 1.0];
    let metric: &dyn Metric<f64, f64> = &MaxError;
    assert_eq!(
        metric.compute(y_true.view(), y_pred.view(), None).unwrap(),
        2.0
    );
}
