use ndarray::{Array1, array};
use predict::PredictError;
use predict::metrics::{self, Average};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

/// Confusion matrix, rows true and columns predicted:
///
/// ```text
///          0  1  2
/// true 0 [ 1  1  0 ]   P = 1/2, R = 1/2, F1 = 1/2
/// true 1 [ 0  2  0 ]   P = 2/3, R = 1,   F1 = 4/5
/// true 2 [ 1  0  2 ]   P = 1,   R = 2/3, F1 = 4/5
/// ```
fn three_classes() -> (Array1<usize>, Array1<usize>) {
    (array![0, 0, 1, 1, 2, 2, 2], array![0, 1, 1, 1, 2, 0, 2])
}

#[test]
fn confusion_matrix_rows_are_true_and_columns_predicted() {
    let (y_true, y_pred) = three_classes();
    let cm = metrics::ConfusionMatrix::from_predictions(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(cm.matrix(), array![[1, 1, 0], [0, 2, 0], [1, 0, 2]]);
    assert_eq!(cm.classes(), &[0, 1, 2]);
    assert_eq!(cm.get(2, 0), Some(1));
    assert_eq!(cm.get(3, 0), None);
}

#[test]
fn confusion_matrix_indexes_only_the_classes_present() {
    let cm = metrics::ConfusionMatrix::from_predictions(array![7, 3].view(), array![3, 3].view())
        .unwrap();
    assert_eq!(cm.classes(), &[3, 7]);
    assert_eq!(cm.matrix(), array![[1, 0], [1, 0]]);
}

#[test]
fn macro_is_the_plain_mean_of_per_class_scores() {
    let (y_true, y_pred) = three_classes();
    let (t, p) = (y_true.view(), y_pred.view());
    let avg = Average::Macro;
    assert_close(
        metrics::precision(t, p, avg).unwrap(),
        (0.5 + 2.0 / 3.0 + 1.0) / 3.0,
    );
    assert_close(
        metrics::recall(t, p, avg).unwrap(),
        (0.5 + 1.0 + 2.0 / 3.0) / 3.0,
    );
    assert_close(
        metrics::f1_score(t, p, avg).unwrap(),
        (0.5 + 0.8 + 0.8) / 3.0,
    );
}

#[test]
fn by_support_weights_each_class_by_its_true_rows() {
    let (y_true, y_pred) = three_classes();
    let (t, p) = (y_true.view(), y_pred.view());
    // Supports 2, 2, 3 of 7 rows.
    let recall = (2.0 * 0.5 + 2.0 * 1.0 + 3.0 * (2.0 / 3.0)) / 7.0;
    assert_close(metrics::recall(t, p, Average::BySupport).unwrap(), recall);
}

#[test]
fn micro_equals_accuracy_with_one_class_per_row() {
    let (y_true, y_pred) = three_classes();
    let (t, p) = (y_true.view(), y_pred.view());
    let accuracy = metrics::accuracy(t, p).unwrap();
    assert_close(metrics::precision(t, p, Average::Micro).unwrap(), accuracy);
    assert_close(metrics::recall(t, p, Average::Micro).unwrap(), accuracy);
    assert_close(metrics::f1_score(t, p, Average::Micro).unwrap(), accuracy);
}

#[test]
fn binary_scores_the_positive_class_only() {
    // Class 1: TP 2, FP 1, FN 1.
    let y_true = array![1, 1, 1, 0, 0];
    let y_pred = array![1, 1, 0, 1, 0];
    let (t, p) = (y_true.view(), y_pred.view());
    let positive = Average::Binary { positive: 1 };
    assert_close(metrics::precision(t, p, positive).unwrap(), 2.0 / 3.0);
    assert_close(metrics::recall(t, p, positive).unwrap(), 2.0 / 3.0);
    let negative = Average::Binary { positive: 0 };
    assert_close(metrics::precision(t, p, negative).unwrap(), 0.5);
}

#[test]
fn binary_rejects_more_than_two_classes_or_an_absent_positive() {
    let (y_true, y_pred) = three_classes();
    let result = metrics::precision(
        y_true.view(),
        y_pred.view(),
        Average::Binary { positive: 1 },
    );
    assert!(matches!(
        result,
        Err(PredictError::InvalidParameter {
            name: "average",
            ..
        })
    ));
    let result = metrics::recall(
        array![0, 1].view(),
        array![0, 1].view(),
        Average::Binary { positive: 5 },
    );
    assert!(matches!(
        result,
        Err(PredictError::InvalidParameter {
            name: "average",
            ..
        })
    ));
}

#[test]
fn binary_with_one_class_and_an_absent_positive_is_zero() {
    let y = array![0, 0];
    let score = metrics::f1_score(y.view(), y.view(), Average::Binary { positive: 1 }).unwrap();
    assert_eq!(score, 0.0);
}

#[test]
fn a_never_predicted_class_scores_zero_and_stays_in_the_macro_mean() {
    // Class 1 is never predicted: precision 0/0 counts as 0.
    let y_true = array![0, 1];
    let y_pred = array![0, 0];
    let precision = metrics::precision(y_true.view(), y_pred.view(), Average::Macro).unwrap();
    // Class 0: 1 / 2; class 1: 0.
    assert_close(precision, 0.25);
}

#[test]
fn fbeta_leans_towards_recall_above_one_and_precision_below() {
    // Class 1: TP 1, FP 1, FN 3, so P = 1/2 and R = 1/4.
    let y_true = array![1, 1, 1, 1, 0];
    let y_pred = array![1, 0, 0, 0, 1];
    let (t, p) = (y_true.view(), y_pred.view());
    let avg = Average::Binary { positive: 1 };
    let fbeta = |beta| metrics::fbeta_score(t, p, beta, avg).unwrap();
    let (precision, recall) = (0.5, 0.25);
    let expected = |b2: f64| (1.0 + b2) * precision * recall / (b2 * precision + recall);
    assert_close(fbeta(2.0), expected(4.0));
    assert_close(fbeta(1.0), metrics::f1_score(t, p, avg).unwrap());
    assert!((fbeta(1e6) - recall).abs() < 1e-6);
    assert!((fbeta(1e-6) - precision).abs() < 1e-6);
}

#[test]
fn fbeta_rejects_a_bad_beta() {
    let y = array![0, 1];
    for beta in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e200] {
        assert!(matches!(
            metrics::fbeta_score(y.view(), y.view(), beta, Average::Macro),
            Err(PredictError::InvalidParameter { name: "beta", .. })
        ));
    }
}

#[test]
fn integer_weights_equal_repeated_rows() {
    let (y_true, y_pred) = three_classes();
    let weights = array![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 1.0];
    let repeated_true = array![0, 0, 0, 1, 2, 2, 2, 2, 2];
    let repeated_pred = array![0, 0, 1, 1, 2, 2, 2, 0, 2];
    let (t, p, w) = (y_true.view(), y_pred.view(), weights.view());
    let (rt, rp) = (repeated_true.view(), repeated_pred.view());
    for avg in [Average::Macro, Average::Micro, Average::BySupport] {
        assert_close(
            metrics::precision_weighted(t, p, avg, w).unwrap(),
            metrics::precision(rt, rp, avg).unwrap(),
        );
        assert_close(
            metrics::recall_weighted(t, p, avg, w).unwrap(),
            metrics::recall(rt, rp, avg).unwrap(),
        );
        assert_close(
            metrics::f1_score_weighted(t, p, avg, w).unwrap(),
            metrics::f1_score(rt, rp, avg).unwrap(),
        );
        assert_close(
            metrics::fbeta_score_weighted(t, p, 0.5, avg, w).unwrap(),
            metrics::fbeta_score(rt, rp, 0.5, avg).unwrap(),
        );
    }
}

#[test]
fn weighted_forms_check_their_weights_and_name_themselves() {
    let (y_true, y_pred) = three_classes();
    let (t, p) = (y_true.view(), y_pred.view());
    assert!(matches!(
        metrics::recall_weighted(t, p, Average::Macro, array![1.0].view()),
        Err(PredictError::ShapeMismatch {
            context: "sample_weights",
            ..
        })
    ));
    let zeros = Array1::<f32>::zeros(7);
    assert!(matches!(
        metrics::f1_score_weighted(t, p, Average::Macro, zeros.view()),
        Err(PredictError::Undefined {
            context: "f1_score_weighted",
            ..
        })
    ));
}

#[test]
fn every_score_checks_lengths_first() {
    let result = metrics::precision(array![0, 1].view(), array![0].view(), Average::Macro);
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "y_pred",
            ..
        })
    ));
    let empty = Array1::<usize>::zeros(0);
    assert!(matches!(
        metrics::ConfusionMatrix::from_predictions(empty.view(), empty.view()),
        Err(PredictError::EmptyInput { context: "y_true" })
    ));
}

#[test]
fn weighted_fbeta_matches_scikit_learn() {
    // fbeta_score(t, p, beta=0.5, average=..., sample_weight=w), scikit-learn 1.x.
    let (y_true, y_pred) = three_classes();
    let weights = array![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 1.0];
    let (t, p, w) = (y_true.view(), y_pred.view(), weights.view());
    let score = |avg| metrics::fbeta_score_weighted(t, p, 0.5, avg, w).unwrap();
    assert_close(score(Average::Macro), 0.724867724867725);
    assert_close(score(Average::Micro), 0.7777777777777778);
    assert_close(score(Average::BySupport), 0.8130511463844797);
}

#[test]
fn accuracy_is_the_fraction_of_correct_classes() {
    let y_true = array![0, 1, 2, 2];
    let y_pred = array![0, 2, 2, 2];
    assert_eq!(
        metrics::accuracy(y_true.view(), y_pred.view()).unwrap(),
        0.75
    );
}

#[test]
fn accuracy_weighted_is_the_share_of_weight_classified_correctly() {
    let y_true = array![0, 1, 2, 2];
    let y_pred = array![0, 2, 2, 2];
    // The wrong row weighs 3 of a total 6.
    let weights = array![1.0_f32, 3.0, 1.0, 1.0];
    let (t, p) = (y_true.view(), y_pred.view());
    assert_eq!(
        metrics::accuracy_weighted(t, p, weights.view()).unwrap(),
        0.5
    );
}

#[test]
fn accuracy_checks_lengths() {
    let result = metrics::accuracy(array![0, 1].view(), array![0].view());
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "y_pred",
            ..
        })
    ));
}
