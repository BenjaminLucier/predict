use ndarray::{Array2, Axis, ShapeBuilder, array};
use predict::Dataset;
use predict::dataset::{
    Absent, BlockView, Blocks, ColMajorMatrix, CscMatrix, CsrMatrix, Finite, MixedDensity,
    MixedOrientation, SparseColMajor, Targets,
};

mod common;
use common::{dense, only_dense};

/// Four rows, one block of each kind side by side:
///
/// ```text
/// row-major | col-major | CSR     | CSC
/// [ 1  2    | 3         | 0  4    | 0  ]
/// [ 5  6    | 7         | 8  0    | 9  ]
/// [ 0  1    | 2         | 0  0    | 3  ]
/// [ 4  5    | 6         | 7  8    | 0  ]
/// ```
fn csr() -> CsrMatrix<'static> {
    CsrMatrix::new(
        2,
        vec![0, 1, 2, 2, 4],
        vec![1, 0, 0, 1],
        vec![4.0, 8.0, 7.0, 8.0],
    )
    .unwrap()
}

fn csc() -> CscMatrix<'static> {
    CscMatrix::new(4, vec![0, 2], vec![1, 2], vec![9.0, 3.0]).unwrap()
}

fn parts() -> (Array2<f64>, Array2<f64>) {
    let mut col_major = Array2::zeros((4, 1).f());
    col_major.assign(&array![[3.0], [7.0], [2.0], [6.0]]);
    (
        array![[1.0, 2.0], [5.0, 6.0], [0.0, 1.0], [4.0, 5.0]],
        col_major,
    )
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

// --- Values ------------------------------------------------------------------------

#[test]
fn every_kind_of_block_keeps_the_chosen_rows_values() {
    let (a, b) = parts();
    let data = every_kind(&a, &b);
    let all = dense(&data);
    let choices: [&[usize]; 6] = [
        &[1, 2],       // a range
        &[0, 1, 2, 3], // everything, a range
        &[3, 0, 2],    // shuffled
        &[2, 2, 0, 2], // repeats, as a bootstrap sample draws
        &[3],          // one row
        &[],           // none
    ];
    for rows in choices {
        let selected = data.select_rows(rows);
        assert_eq!(selected.n_samples(), rows.len(), "{rows:?}");
        assert_eq!(selected.n_features(), 6, "{rows:?}");
        assert_eq!(dense(&selected), all.select(Axis(0), rows), "{rows:?}");
    }
}

#[test]
fn every_block_keeps_its_kind() {
    let (a, b) = parts();
    let data = every_kind(&a, &b);
    let kinds = |data: &Dataset<'_, f64, Blocks<'_, f64, MixedOrientation, MixedDensity>>| {
        data.blocks()
            .map(|block| match block {
                BlockView::DenseRowMajor(x) => ("row-major", x.is_standard_layout()),
                BlockView::DenseColMajor(x) => ("col-major", x.t().is_standard_layout()),
                BlockView::Csr(_) => ("csr", true),
                BlockView::Csc(_) => ("csc", true),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
    };
    let expected = kinds(&data);
    assert_eq!(kinds(&data.select_rows(&[3, 1])), expected);
    assert_eq!(kinds(&data.select_rows(&[1, 2])), expected);
}

#[test]
fn a_selected_csc_block_is_canonical_even_with_repeats_out_of_order() {
    // Rebuilding the selected parts through the checked constructor proves the
    // row indices of each column are in range and strictly increasing.
    let data = Dataset::from_csc(csc());
    let selected = data.select_rows(&[2, 1, 2, 1]);
    let Some(BlockView::Csc(m)) = selected.blocks().next() else {
        panic!("expected a CSC block");
    };
    let rebuilt = CscMatrix::new(m.n_rows(), m.col_offsets(), m.row_indices(), m.values());
    assert!(rebuilt.is_ok(), "{rebuilt:?}");
    assert_eq!(m.row_indices(), &[0, 1, 2, 3]);
    assert_eq!(m.values(), &[3.0, 9.0, 3.0, 9.0]);
}

#[test]
fn sparse_blocks_keep_what_absent_entries_mean() {
    let data = Dataset::from_csr(csr().with_absent(Absent::Missing));
    let selected = data.select_rows(&[2, 0]);
    // Row 2 stores nothing, so it is all missing.
    assert!(dense(&selected).row(0).iter().all(|v| v.is_nan()));
}

// --- Borrowed or copied --------------------------------------------------------------

#[test]
fn a_range_of_rows_is_borrowed_not_copied() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (y, w) = (array![0_usize, 1, 0], array![1.0, 2.0, 3.0]);
    let data = Dataset::new(&x)
        .with_y(&y)
        .unwrap()
        .with_sample_weights(&w)
        .unwrap();

    let tail = data.select_rows(&[1, 2]);
    assert_eq!(only_dense(&tail).as_ptr(), x.row(1).as_ptr());
    assert_eq!(tail.y().as_ptr(), &y[1] as *const usize);
    assert_eq!(tail.sample_weights().unwrap().as_ptr(), &w[1] as *const f64);

    // Shuffled rows are copied.
    let shuffled = data.select_rows(&[2, 1]);
    assert_ne!(only_dense(&shuffled).as_ptr(), x.row(2).as_ptr());
    assert_eq!(shuffled.y(), array![0, 1]);
}

#[test]
fn a_range_of_col_major_rows_is_borrowed_and_stays_col_major() {
    let mut f = Array2::<f64>::zeros((4, 2).f());
    f.assign(&array![[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]);
    let data = Dataset::from_col_major(ColMajorMatrix::new(&f).unwrap());
    let middle = data.select_rows(&[1, 2]);
    let view = only_dense(&middle);
    assert_eq!(view.as_ptr(), &f[[1, 0]] as *const f64);
    assert_eq!(view, array![[2.0, 6.0], [3.0, 7.0]]);
    assert!(view.column(0).as_slice().is_some());
}

#[test]
fn a_range_of_csr_rows_borrows_its_entries_and_rebuilds_only_offsets() {
    let m = csr();
    let data = Dataset::from_csr(m.clone());
    let Some(BlockView::Csr(original)) = data.blocks().next() else {
        unreachable!()
    };
    let tail = data.select_rows(&[1, 2, 3]);
    let Some(BlockView::Csr(selected)) = tail.blocks().next() else {
        unreachable!()
    };
    assert_eq!(selected.row_offsets(), &[0, 1, 1, 3]);
    assert_eq!(
        selected.col_indices().as_ptr(),
        original.col_indices()[1..].as_ptr()
    );
    assert_eq!(selected.values().as_ptr(), original.values()[1..].as_ptr());
}

// --- Types, state, targets ----------------------------------------------------------

#[test]
fn the_state_layout_and_targets_keep_their_types() {
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![1.5, 2.5, 3.5];
    let data = Dataset::from_csc(
        CscMatrix::new(3, vec![0, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap(),
    )
    .validate_finite()
    .unwrap()
    .with_y(&y)
    .unwrap();
    let selected: Dataset<'_, f64, SparseColMajor<'_>, Finite, Targets<'_, f64>> =
        data.select_rows(&[2, 0]);
    assert_eq!(selected.y(), array![3.5, 1.5]);
    assert_eq!(dense(&selected), x.select(Axis(0), &[2, 0]));
}

#[test]
#[should_panic(expected = "select_rows: row 3 is out of range for 3 rows")]
fn a_row_out_of_range_panics() {
    let x = array![[1.0], [2.0], [3.0]];
    let _ = Dataset::new(&x).select_rows(&[0, 3]);
}
