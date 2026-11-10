use ndarray::{Array2, array};
use predict::dataset::{Absent, CsrMatrix, Finite};
use predict::{Dataset, PredictError};

/// The 3 × 4 example from the docs:
///
/// ```text
/// [ 5  0  0  1 ]
/// [ 0  0  2  0 ]
/// [ 0  3  0  0 ]
/// ```
fn example() -> CsrMatrix<'static> {
    CsrMatrix::new(
        4,
        vec![0, 2, 3, 4],
        vec![0, 3, 2, 1],
        vec![5.0, 1.0, 2.0, 3.0],
    )
    .unwrap()
}

// --- Reading -----------------------------------------------------------------

#[test]
fn the_example_reads_back_as_its_dense_matrix() {
    let m = example();
    assert_eq!((m.n_rows(), m.n_cols(), m.n_stored()), (3, 4, 4));
    assert_eq!(m.absent(), Absent::Zero);

    let data = Dataset::from_csr(m);
    assert_eq!((data.n_samples(), data.n_features()), (3, 4));
    let dense = data.to_dense();
    assert!(dense.is_owned());
    assert_eq!(
        dense,
        array![
            [5.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0]
        ]
    );
}

#[test]
fn missing_absent_entries_read_as_nan() {
    // (Absent zeros are checked by the example above.)
    let data = Dataset::from_csr(example().with_absent(Absent::Missing));
    let dense = data.to_dense();
    assert_eq!(dense[[0, 0]], 5.0);
    assert!(dense[[0, 1]].is_nan());
    assert_eq!(dense.iter().filter(|value| value.is_nan()).count(), 12 - 4);
}

// --- Borrowed or owned, never copied -------------------------------------------

#[test]
fn the_arrays_are_borrowed_or_moved_without_copying() {
    let (offsets, cols, values) = (vec![0, 1], vec![2], vec![7.0]);

    // Borrowed: the matrix reads the caller's arrays in place.
    let m = CsrMatrix::new(3, &offsets, &cols[..], &values).unwrap();
    assert_eq!(m.row_offsets().as_ptr(), offsets.as_ptr());
    assert_eq!(m.col_indices().as_ptr(), cols.as_ptr());
    assert_eq!(m.values().as_ptr(), values.as_ptr());

    // Moved in: the matrix owns the same buffer.
    let ptr = values.as_ptr();
    let m = CsrMatrix::new(3, offsets, cols, values).unwrap();
    assert_eq!(m.values().as_ptr(), ptr);
}

#[test]
fn an_owned_sparse_dataset_outlives_the_function_that_made_it() {
    fn make() -> Dataset<'static> {
        Dataset::from_csr(example())
            .with_sample_weights(array![1.0, 2.0, 1.0])
            .unwrap()
    }
    assert_eq!(make().n_samples(), 3);
}

// --- Structure -----------------------------------------------------------------

#[derive(Debug)]
enum Expect {
    Ok,
    Malformed(&'static str),
    ShapeMismatch(&'static str, usize, usize),
}

/// (case, n_cols, row_offsets, col_indices, n_values, expected)
type StructureCase = (
    &'static str,
    usize,
    &'static [usize],
    &'static [usize],
    usize,
    Expect,
);

#[test]
fn the_structure_is_checked_offsets_then_lengths_then_columns() {
    use Expect::*;
    let cases: [StructureCase; 14] = [
        ("the example", 4, &[0, 2, 3, 4], &[0, 3, 2, 1], 4, Ok),
        ("no rows", 4, &[0], &[], 0, Ok),
        ("no columns", 0, &[0, 0, 0], &[], 0, Ok),
        (
            "rows and trailing columns with nothing stored",
            6,
            &[0, 0, 1, 1],
            &[2],
            1,
            Ok,
        ),
        (
            "no offsets at all",
            4,
            &[],
            &[],
            0,
            Malformed("row_offsets"),
        ),
        (
            "offsets not starting at 0",
            4,
            &[1, 2],
            &[0],
            1,
            Malformed("row_offsets"),
        ),
        (
            "offsets decreasing",
            4,
            &[0, 2, 1, 3],
            &[0, 1, 2],
            3,
            Malformed("row_offsets"),
        ),
        (
            "too few column indices",
            4,
            &[0, 2],
            &[0],
            2,
            ShapeMismatch("col_indices", 2, 1),
        ),
        (
            "too many values",
            4,
            &[0, 2],
            &[0, 1],
            3,
            ShapeMismatch("values", 2, 3),
        ),
        (
            "column out of range",
            4,
            &[0, 1, 2],
            &[0, 4],
            2,
            Malformed("col_indices"),
        ),
        (
            "columns out of order",
            4,
            &[0, 2],
            &[3, 0],
            2,
            Malformed("col_indices"),
        ),
        (
            "a column repeated",
            4,
            &[0, 2],
            &[2, 2],
            2,
            Malformed("col_indices"),
        ),
        (
            "offsets are checked before lengths",
            4,
            &[0, 2, 1],
            &[0],
            5,
            Malformed("row_offsets"),
        ),
        (
            "lengths are checked before columns",
            4,
            &[0, 2],
            &[9],
            2,
            ShapeMismatch("col_indices", 2, 1),
        ),
    ];

    for (case, n_cols, offsets, cols, n_values, expected) in cases {
        let values = vec![1.0; n_values];
        let result = CsrMatrix::new(n_cols, offsets, cols, values).map(|_| ());
        match (result, &expected) {
            (Result::Ok(()), Ok) => {}
            (Err(PredictError::Malformed { context, .. }), Malformed(expected)) => {
                assert_eq!(context, *expected, "{case}")
            }
            (
                Err(PredictError::ShapeMismatch {
                    context,
                    expected,
                    actual,
                }),
                ShapeMismatch(c, e, a),
            ) => assert_eq!((context, expected, actual), (*c, *e, *a), "{case}"),
            (result, _) => panic!("{case}: expected {expected:?}, got {result:?}"),
        }
    }
}

#[test]
fn a_malformed_row_is_named_in_the_error() {
    let err = CsrMatrix::new(4, vec![0, 1, 3], vec![0, 3, 3], vec![1.0; 3]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "malformed col_indices: row 1: column 3 does not come after column 3"
    );
}

// --- Validation -------------------------------------------------------------------

#[test]
fn validation_reads_stored_values_at_their_position() {
    // Row 1 stores NaN at column 2; row 2 stores infinity at column 0.
    let m = CsrMatrix::new(
        4,
        vec![0, 1, 2, 3],
        vec![0, 2, 0],
        vec![1.0, f64::NAN, f64::INFINITY],
    )
    .unwrap();

    let err = Dataset::from_csr(m.clone()).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 1, col: 2, .. }
    ));

    // NaN is a missing value, which `NoInfinity` allows; the infinity is not.
    let err = Dataset::from_csr(m).validate_no_infinity().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 2, col: 0, .. }
    ));

    // Absent zeros are finite.
    let finite: Dataset<'_, _, _, Finite> = Dataset::from_csr(example()).validate_finite().unwrap();
    assert_eq!(finite.n_samples(), 3);
}

#[test]
fn missing_absent_entries_are_allowed_until_finite_is_required() {
    // Row 0 stores columns 0 and 3, so column 1 is its first absent entry.
    let missing = example().with_absent(Absent::Missing);
    assert!(
        Dataset::from_csr(missing.clone())
            .validate_no_infinity()
            .is_ok()
    );
    let err = Dataset::from_csr(missing).validate_finite().unwrap_err();
    assert!(matches!(
        err,
        PredictError::NonFinite { row: 0, col: 1, value, .. } if value.is_nan()
    ));

    // Every entry stored: nothing is missing, so it can be `Finite`.
    let full = CsrMatrix::new(2, vec![0, 2], vec![0, 1], vec![1.0, 2.0])
        .unwrap()
        .with_absent(Absent::Missing);
    assert!(Dataset::from_csr(full).validate_finite().is_ok());
}

#[test]
fn within_a_row_the_lower_column_is_reported_stored_or_absent() {
    // (case, stored columns, stored values, expected column) in a 1 × 4 row.
    let cases: [(&str, &[usize], &[f64], usize); 4] = [
        (
            "stored NaN before the first gap",
            &[0, 1],
            &[f64::NAN, 1.0],
            0,
        ),
        (
            "the first gap before a stored NaN",
            &[0, 3],
            &[1.0, f64::NAN],
            1,
        ),
        ("a gap at column 0", &[1, 2, 3], &[1.0, 1.0, 1.0], 0),
        ("a gap only at the end", &[0, 1, 2], &[1.0, 1.0, 1.0], 3),
    ];
    for (case, cols, values, expected) in cases {
        let m = CsrMatrix::new(4, vec![0, cols.len()], cols, values)
            .unwrap()
            .with_absent(Absent::Missing);
        match Dataset::from_csr(m).validate_finite() {
            Err(PredictError::NonFinite { row: 0, col, .. }) => assert_eq!(col, expected, "{case}"),
            other => panic!("{case}: {other:?}"),
        }
    }
}

// --- With targets and weights ------------------------------------------------------

#[test]
fn targets_and_weights_have_one_row_per_sparse_row() {
    let data = Dataset::from_csr(example());
    let err = data.clone().with_y(array![1.0, 2.0]).unwrap_err();
    assert!(matches!(
        err,
        PredictError::ShapeMismatch {
            context: "y",
            expected: 3,
            actual: 2
        }
    ));

    let data = data
        .with_y(array![0_usize, 1, 0])
        .unwrap()
        .with_sample_weights(array![1.0, 1.0, 2.0])
        .unwrap()
        .validate_finite()
        .unwrap();
    assert_eq!(data.y().len(), 3);
    let _: Array2<f64> = data.to_dense().into_owned();
}
