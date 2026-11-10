use ndarray::{Array2, array};
use predict::dataset::{Absent, CscMatrix, CsrMatrix, SparseColMajor};
use predict::{Dataset, PredictError};

/// The 3 × 4 example from the docs, column by column:
///
/// ```text
/// [ 5  0  0  1 ]
/// [ 0  0  2  0 ]
/// [ 0  3  0  0 ]
/// ```
fn example() -> CscMatrix<'static> {
    CscMatrix::new(
        3,
        vec![0, 1, 2, 3, 4],
        vec![0, 2, 1, 0],
        vec![5.0, 3.0, 2.0, 1.0],
    )
    .unwrap()
}

// --- Reading -----------------------------------------------------------------

#[test]
fn the_example_reads_back_as_its_dense_matrix() {
    let m = example();
    assert_eq!((m.n_rows(), m.n_cols(), m.n_stored()), (3, 4, 4));
    assert_eq!(m.absent(), Absent::Zero);

    let data: Dataset<'_, f64, SparseColMajor<'_>> = Dataset::from_csc(m);
    assert_eq!((data.n_samples(), data.n_features()), (3, 4));
    assert_eq!(
        data.to_dense(),
        array![
            [5.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0]
        ]
    );
}

#[test]
fn csc_and_csr_of_the_same_matrix_read_the_same() {
    let csr = CsrMatrix::new(
        4,
        vec![0, 2, 3, 4],
        vec![0, 3, 2, 1],
        vec![5.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    assert_eq!(
        Dataset::from_csc(example()).to_dense(),
        Dataset::from_csr(csr).to_dense()
    );
}

#[test]
fn missing_absent_entries_read_as_nan() {
    let data = Dataset::from_csc(example().with_absent(Absent::Missing));
    let dense = data.to_dense();
    assert_eq!(dense[[0, 0]], 5.0);
    assert!(dense[[1, 0]].is_nan());
}

#[test]
fn the_arrays_are_borrowed_or_moved_without_copying() {
    let (offsets, rows, values) = (vec![0, 1], vec![0], vec![4.0]);
    let m = CscMatrix::new(2, &offsets, &rows, &values).unwrap();
    assert_eq!(m.values().as_ptr(), values.as_ptr());
    assert_eq!(m.row_indices().as_ptr(), rows.as_ptr());
    assert_eq!(m.col_offsets().as_ptr(), offsets.as_ptr());
}

// --- Structure -----------------------------------------------------------------

fn malformed(n_rows: usize, offsets: Vec<usize>, rows: Vec<usize>) -> PredictError {
    let values = vec![1.0; rows.len()];
    CscMatrix::new(n_rows, offsets, rows, values).unwrap_err()
}

#[test]
fn the_structure_is_checked_with_csc_names() {
    let err = malformed(2, vec![1, 1], vec![]);
    assert_eq!(
        err.to_string(),
        "malformed col_offsets: must start at 0, got 1"
    );

    let err = CscMatrix::new(2, vec![0, 2], vec![0], vec![1.0, 1.0]).unwrap_err();
    assert!(matches!(
        err,
        PredictError::ShapeMismatch {
            context: "row_indices",
            expected: 2,
            actual: 1
        }
    ));

    let err = malformed(2, vec![0, 1], vec![2]);
    assert_eq!(
        err.to_string(),
        "malformed row_indices: column 0: row 2 is out of range for 2 rows"
    );

    let err = malformed(3, vec![0, 2], vec![1, 0]);
    assert_eq!(
        err.to_string(),
        "malformed row_indices: column 0: row 0 does not come after row 1"
    );
}

// --- Validation ------------------------------------------------------------------

#[test]
fn bad_values_are_reported_at_their_row_and_column() {
    // Column 1 is scanned before column 3, so (2, 1) is found before (0, 3),
    // although (0, 3) comes first reading row by row.
    let m = CscMatrix::new(
        3,
        vec![0, 0, 1, 1, 2],
        vec![2, 0],
        vec![f64::NAN, f64::INFINITY],
    )
    .unwrap();
    let err = Dataset::from_csc(m).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 2, col: 1, .. }
    ));
}

#[test]
fn missing_absent_entries_are_allowed_until_finite_is_required() {
    let missing = || Dataset::from_csc(example().with_absent(Absent::Missing));
    assert!(missing().validate_no_infinity().is_ok());
    // Column 0 stores only row 0, so (1, 0) is the first absent entry.
    let err = missing().validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 1, col: 0, .. }
    ));
}

#[test]
fn targets_have_one_row_per_sparse_row() {
    let err = Dataset::from_csc(example())
        .with_y(array![1.0, 2.0])
        .unwrap_err();
    assert!(matches!(
        err,
        PredictError::ShapeMismatch {
            context: "y",
            expected: 3,
            actual: 2
        }
    ));
    let empty = CscMatrix::<f64>::new(0, vec![0], vec![], vec![]).unwrap();
    assert_eq!(
        Dataset::from_csc(empty).to_dense(),
        Array2::<f64>::zeros((0, 0))
    );
}
