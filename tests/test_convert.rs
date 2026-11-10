use ndarray::{Array2, ShapeBuilder, array};
use predict::dataset::{
    Absent, BlockView, Blocks, ColMajorMatrix, CscMatrix, CsrMatrix, DenseColMajor, DenseRowMajor,
    Finite, MixedDensity, MixedOrientation, SparseColMajor, SparseRowMajor, State,
};
use predict::{Dataset, PredictError};

mod common;
use common::{dense, only_dense};

/// Two rows with every kind of block side by side:
///
/// ```text
/// row-major | col-major | CSR        | CSC
/// [ 1  2    | 3         | 0  4       | 0  ]
/// [ 5  0    | 6         | 7  0       | 8  ]
/// ```
fn csr() -> CsrMatrix<'static> {
    CsrMatrix::new(2, vec![0, 1, 2], vec![1, 0], vec![4.0, 7.0]).unwrap()
}

fn csc() -> CscMatrix<'static> {
    CscMatrix::new(2, vec![0, 1], vec![1], vec![8.0]).unwrap()
}

fn expected() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0, 0.0, 4.0, 0.0],
        [5.0, 0.0, 6.0, 7.0, 0.0, 8.0]
    ]
}

fn every_kind<'a>(
    row_major: &'a Array2<f64>,
    col_major: &'a Array2<f64>,
) -> Dataset<'a, f64, Blocks<'a, f64, MixedOrientation, MixedDensity>> {
    Dataset::new(row_major)
        .hstack(Dataset::from_col_major(
            ColMajorMatrix::new(col_major).unwrap(),
        ))
        .unwrap()
        .hstack(Dataset::from_csr(csr()))
        .unwrap()
        .hstack(Dataset::from_csc(csc()))
        .unwrap()
}

fn parts() -> (Array2<f64>, Array2<f64>) {
    let mut col_major = Array2::zeros((2, 1).f());
    col_major.assign(&array![[3.0], [6.0]]);
    (array![[1.0, 2.0], [5.0, 0.0]], col_major)
}

// --- Every conversion keeps every value -------------------------------------------

#[test]
fn every_conversion_from_every_kind_keeps_the_values() {
    let (a, b) = parts();
    let rows: Dataset<'_, f64, DenseRowMajor<'_>> = every_kind(&a, &b).into_dense_row_major();
    assert_eq!(only_dense(&rows), expected());
    assert!(only_dense(&rows).is_standard_layout());

    let cols: Dataset<'_, f64, DenseColMajor<'_>> = every_kind(&a, &b).into_dense_col_major();
    assert_eq!(only_dense(&cols), expected());
    assert!(only_dense(&cols).t().is_standard_layout());

    let csr: Dataset<'_, f64, SparseRowMajor<'_>> = every_kind(&a, &b).into_csr();
    assert_eq!(dense(&csr), expected());

    let csc: Dataset<'_, f64, SparseColMajor<'_>> = every_kind(&a, &b).into_csc();
    assert_eq!(dense(&csc), expected());
}

#[test]
fn converting_back_and_forth_keeps_the_values() {
    let (a, b) = parts();
    let round_trip = every_kind(&a, &b)
        .into_csc()
        .into_dense_col_major()
        .into_csr()
        .into_dense_row_major();
    assert_eq!(dense(&round_trip), expected());
}

#[test]
fn one_block_of_the_target_kind_is_moved_not_copied() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let same = Dataset::new(&x).into_dense_row_major();
    assert_eq!(only_dense(&same).as_ptr(), x.as_ptr());

    let f = Array2::<f64>::zeros((2, 2).f());
    let same = Dataset::from_col_major(ColMajorMatrix::new(&f).unwrap()).into_dense_col_major();
    assert_eq!(only_dense(&same).as_ptr(), f.as_ptr());

    // The other order is a copy.
    let copied = Dataset::new(&x).into_dense_col_major();
    assert_ne!(only_dense(&copied).as_ptr(), x.as_ptr());
}

// --- What absent entries mean -------------------------------------------------------

#[test]
fn dense_to_sparse_leaves_out_what_absent_stands_for() {
    // With `Zero`, the zeros are left out and NaN is stored; with `Missing`,
    // NaN is left out and the zeros are stored. Either way the values are the
    // same, and validation finds the NaN at (0, 1).
    let x = array![[0.0, f64::NAN], [2.0, 0.0]];
    for absent in [Absent::Zero, Absent::Missing] {
        let sparse = Dataset::new(&x).into_csr_with(absent);
        let dense = dense(&sparse);
        assert_eq!(
            (dense[[0, 0]], dense[[1, 0]], dense[[1, 1]]),
            (0.0, 2.0, 0.0)
        );
        assert!(dense[[0, 1]].is_nan());
        let err = sparse.validate_finite().unwrap_err();
        assert!(matches!(
            err,
            PredictError::NonFinite { row: 0, col: 1, .. }
        ));
    }
}

/// The single CSR block's `(row offsets, column indices)`.
fn stored<'a, S: State>(
    data: &Dataset<'a, f64, SparseRowMajor<'a>, S>,
) -> (Vec<usize>, Vec<usize>) {
    match data.blocks().next() {
        Some(BlockView::Csr(m)) => (m.row_offsets().to_vec(), m.col_indices().to_vec()),
        _ => panic!("expected one CSR block"),
    }
}

#[test]
fn dense_to_sparse_stores_exactly_what_absent_does_not_stand_for() {
    let x = array![[0.0, f64::NAN], [2.0, 0.0]];
    // `Zero`: the NaN at (0, 1) and the 2 at (1, 0).
    let zero = Dataset::new(&x).into_csr_with(Absent::Zero);
    assert_eq!(stored(&zero), (vec![0, 1, 2], vec![1, 0]));
    // `Missing`: the 0 at (0, 0), then the 2 and the 0 in row 1.
    let missing = Dataset::new(&x).into_csr_with(Absent::Missing);
    assert_eq!(stored(&missing), (vec![0, 1, 3], vec![0, 0, 1]));
}

#[test]
fn several_blocks_merge_into_one_csr_with_sorted_columns() {
    let (a, b) = parts();
    let csr = every_kind(&a, &b).into_csr();
    assert_eq!(csr.blocks().len(), 1);
    // Row 0 stores columns 0, 1, 2, 4; row 1 stores 0, 2, 3, 5.
    assert_eq!(stored(&csr), (vec![0, 4, 8], vec![0, 1, 2, 4, 0, 2, 3, 5]));
}

#[test]
fn a_new_meaning_keeps_the_values_a_sparse_block_left_out() {
    // `csr()` leaves out zeros. Converted to `Missing`, those zeros must be
    // stored, so the matrix is still finite.
    let data = Dataset::from_csr(csr()).into_csc_with(Absent::Missing);
    assert_eq!(dense(&data), array![[0.0, 4.0], [7.0, 0.0]]);
    assert!(data.validate_finite().is_ok());
}

#[test]
fn into_csr_keeps_the_datasets_meaning() {
    let missing = Dataset::from_csc(csc().with_absent(Absent::Missing)).into_csr();
    let dense = dense(&missing);
    assert!(dense[[0, 0]].is_nan());
    assert_eq!(dense[[1, 0]], 8.0);
}

#[test]
fn with_absent_changes_what_the_left_out_entries_stand_for() {
    let data = Dataset::from_csr(csr()).with_absent(Absent::Missing);
    let dense = dense(&data);
    assert!(dense[[0, 0]].is_nan());
    assert_eq!(dense[[0, 1]], 4.0);
}

// --- What is kept -------------------------------------------------------------------

#[test]
fn conversions_keep_the_state_targets_and_weights() {
    let (a, b) = parts();
    let (y, w) = (array![1.0, 2.0], array![1.0, 3.0]);
    let data = every_kind(&a, &b)
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap();
    let csc = data.into_csc();
    let _: &Dataset<'_, f64, SparseColMajor<'_>, Finite, _> = &csc;
    assert_eq!(csc.y(), y);
    assert_eq!(csc.sample_weights().unwrap(), w);
}
