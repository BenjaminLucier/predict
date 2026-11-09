use ndarray::{Array2, array};
use predict::dataset::{Absent, CsrMatrix, Finite, NoInfinity};
use predict::{Dataset, PredictError};

/// A 2 × 3 sparse matrix storing 7 at (0, 2) and 8 at (1, 0).
fn sparse() -> CsrMatrix<'static> {
    CsrMatrix::new(3, vec![0, 1, 2], vec![2, 0], vec![7.0, 8.0]).unwrap()
}

// --- Reading -----------------------------------------------------------------

#[test]
fn several_dense_blocks_read_as_one_matrix() {
    let (a, b) = (array![[1.0], [2.0]], array![[3.0, 4.0], [5.0, 6.0]]);
    let data = Dataset::new(&a).hstack(Dataset::new(&b)).unwrap();
    assert_eq!((data.n_samples(), data.n_features()), (2, 3));

    let dense = data.to_dense();
    assert!(dense.is_owned());
    assert_eq!(dense, array![[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]);
}

#[test]
fn dense_and_sparse_read_as_one_matrix() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let data = Dataset::new(&x)
        .hstack(Dataset::from_csr(sparse()))
        .unwrap();
    assert_eq!((data.n_samples(), data.n_features()), (2, 5));
    assert_eq!(
        data.to_dense(),
        array![[1.0, 2.0, 0.0, 0.0, 7.0], [3.0, 4.0, 8.0, 0.0, 0.0]]
    );
}

#[test]
fn stacking_keeps_column_order_however_it_is_grouped() {
    let (a, b, c) = (array![[1.0]], array![[2.0]], array![[3.0]]);
    let left_first = Dataset::new(&a)
        .hstack(Dataset::new(&b))
        .unwrap()
        .hstack(Dataset::new(&c))
        .unwrap();
    let right_first = Dataset::new(&a)
        .hstack(Dataset::new(&b).hstack(Dataset::new(&c)).unwrap())
        .unwrap();
    assert_eq!(left_first.to_dense(), array![[1.0, 2.0, 3.0]]);
    assert_eq!(right_first.to_dense(), left_first.to_dense());
}

#[test]
fn an_owned_stacked_dataset_outlives_the_function_that_made_it() {
    fn make() -> Dataset<'static> {
        Dataset::new(array![[1.0], [2.0]])
            .hstack(Dataset::from_csr(sparse()))
            .unwrap()
    }
    assert_eq!(make().n_features(), 4);
}

// --- Validation ------------------------------------------------------------------

#[test]
fn a_bad_value_is_reported_at_its_column_in_the_dataset() {
    // The NaN is in column 1 of the second block, so column 2 of the dataset.
    let (a, b) = (array![[1.0], [2.0]], array![[3.0, 4.0], [5.0, f64::NAN]]);
    let stacked = |a| Dataset::new(a).hstack(Dataset::new(&b)).unwrap();
    let err = stacked(&a).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 1, col: 2, .. }
    ));

    // Blocks are scanned in column order: the first block's bad value wins.
    let a = array![[f64::INFINITY], [2.0]];
    let err = stacked(&a).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 0, col: 0, .. }
    ));
}

#[test]
fn a_missing_sparse_entry_is_reported_at_its_column_in_the_dataset() {
    // Row 0 of the sparse block stores only column 2, so its column 0 (dataset
    // column 2) is the first missing entry.
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let missing = Dataset::from_csr(sparse().with_absent(Absent::Missing));
    let err = Dataset::new(&x)
        .hstack(missing)
        .unwrap()
        .validate_finite()
        .unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 0, col: 2, .. }
    ));
}

#[test]
fn stacking_keeps_the_state_of_both_sides() {
    let (a, b) = (array![[1.0], [2.0]], array![[f64::NAN], [3.0]]);
    let finite = |x| Dataset::new(x).validate_finite().unwrap();
    let _: Dataset<'_, f64, Finite> = finite(&a).hstack(finite(&a)).unwrap();

    let no_infinity = |x| Dataset::new(x).validate_no_infinity().unwrap();
    let _: Dataset<'_, f64, NoInfinity> = no_infinity(&a).hstack(no_infinity(&b)).unwrap();
}

// --- Targets and weights -----------------------------------------------------------

#[test]
fn the_left_sides_targets_and_weights_are_kept() {
    let (x, y, w) = (array![[1.0], [2.0]], array![0_usize, 1], array![1.0, 3.0]);
    let data = Dataset::new(&x)
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap()
        .hstack(Dataset::from_csr(sparse()))
        .unwrap();
    assert_eq!(data.y(), y);
    assert_eq!(data.sample_weights().unwrap(), w);
    assert_eq!(data.n_features(), 4);
}

// --- Errors ------------------------------------------------------------------------

#[test]
fn both_sides_need_the_same_number_of_rows() {
    let (a, b) = (Array2::<f64>::zeros((3, 1)), Array2::<f64>::zeros((2, 1)));
    let err = Dataset::new(&a).hstack(Dataset::new(&b)).unwrap_err();
    assert!(matches!(
        err,
        PredictError::ShapeMismatch {
            context: "hstack",
            expected: 3,
            actual: 2
        }
    ));
}

#[test]
fn weights_on_the_right_side_conflict() {
    let x = array![[1.0], [2.0]];
    let weighted = Dataset::new(&x)
        .with_sample_weights(array![1.0, 2.0])
        .unwrap();
    let err = Dataset::new(&x).hstack(weighted).unwrap_err();
    assert_eq!(
        err.to_string(),
        "hstack: the right side has sample weights; attach weights after stacking"
    );
    assert!(matches!(
        err,
        PredictError::Conflict {
            context: "hstack",
            ..
        }
    ));
}

#[test]
fn rows_are_checked_before_weights() {
    let (a, b) = (Array2::<f64>::zeros((3, 1)), Array2::<f64>::zeros((2, 1)));
    let weighted = Dataset::new(&b)
        .with_sample_weights(array![1.0, 1.0])
        .unwrap();
    let err = Dataset::new(&a).hstack(weighted).unwrap_err();
    assert!(matches!(err, PredictError::ShapeMismatch { .. }));
}
