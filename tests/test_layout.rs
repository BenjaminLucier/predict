use ndarray::{Array2, ShapeBuilder, array, s};
use predict::Dataset;
use predict::PredictError;
use predict::dataset::{
    Blocks, ColMajorMatrix, Cols, CscMatrix, CsrMatrix, Dense, DenseColMajor, DenseRowMajor,
    MixedDensity, MixedOrientation, RowMajorMatrix, Rows, Sparse, SparseColMajor, SparseRowMajor,
    Union,
};

mod common;
use common::{dense, only_dense};

/// A 2 × 3 sparse matrix storing 7 at (0, 2) and 8 at (1, 0).
fn sparse() -> CsrMatrix<'static> {
    CsrMatrix::new(3, vec![0, 1, 2], vec![2, 0], vec![7.0, 8.0]).unwrap()
}

// --- Constructors --------------------------------------------------------------

#[test]
fn new_and_from_csr_name_their_layout() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let _: Dataset<'_, f64, DenseRowMajor<'_>> = Dataset::new(&x);
    let _: Dataset<'_, f64, SparseRowMajor<'_>> = Dataset::from_csr(sparse());
}

#[test]
fn each_constructor_names_its_layout() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let f = Array2::<f64>::zeros((2, 2).f());
    let csc = CscMatrix::new(2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap();
    let _: Dataset<'_, f64, DenseRowMajor<'_>> =
        Dataset::from_row_major(RowMajorMatrix::new(&x).unwrap());
    let _: Dataset<'_, f64, DenseColMajor<'_>> =
        Dataset::from_col_major(ColMajorMatrix::new(&f).unwrap());
    let _: Dataset<'_, f64, SparseColMajor<'_>> = Dataset::from_csc(csc);
}

#[test]
fn the_parse_types_return_the_error_new_panics_with() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let err = RowMajorMatrix::new(x.t()).unwrap_err();
    assert!(matches!(err, PredictError::Malformed { context: "x", .. }));
    let err = ColMajorMatrix::new(&x).unwrap_err();
    assert!(err.to_string().contains("the columns are not contiguous"));
    // The transpose of a row-major array is column-major, as a different matrix.
    let t = Dataset::from_col_major(ColMajorMatrix::new(x.t()).unwrap());
    assert_eq!(dense(&t), array![[1.0, 3.0], [2.0, 4.0]]);
    assert_eq!(only_dense(&t).as_ptr(), x.as_ptr());
}

#[test]
fn new_borrows_a_view_of_some_columns_without_copying() {
    // Each row of the view is contiguous; the next row starts 3 values on, not
    // 2, which a row-major layout allows.
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let view = x.slice(s![.., 1..]);
    let data = Dataset::new(view);
    assert_eq!(dense(&data), array![[2.0, 3.0], [5.0, 6.0]]);
    assert_eq!(only_dense(&data).as_ptr(), view.as_ptr());
}

#[test]
fn a_single_column_or_row_is_contiguous_either_way() {
    let column = array![[1.0], [2.0], [3.0]];
    assert_eq!(Dataset::new(column.t()).n_features(), 3);
    let fortran = Array2::<f64>::zeros((1, 4).f());
    assert_eq!(Dataset::new(&fortran).n_features(), 4);
}

#[test]
#[should_panic(expected = "the rows are not contiguous")]
fn new_refuses_a_column_major_array() {
    let x = Array2::<f64>::zeros((3, 2).f());
    let _ = Dataset::new(&x);
}

#[test]
#[should_panic(expected = "as_standard_layout")]
fn new_refuses_a_transposed_view() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let _ = Dataset::new(x.t());
}

#[test]
#[should_panic(expected = "the rows are not contiguous")]
fn new_refuses_a_view_that_skips_columns() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let _ = Dataset::new(x.slice(s![.., ..;2]));
}

#[test]
fn an_explicit_copy_into_row_major_order_is_accepted() {
    let x = Array2::from_shape_vec((2, 2).f(), vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    let data = Dataset::new(x.as_standard_layout());
    assert_eq!(dense(&data), array![[1.0, 2.0], [3.0, 4.0]]);
}

// --- Stacking ------------------------------------------------------------------

#[test]
fn stacking_keeps_a_kind_both_sides_share() {
    let (a, b) = (array![[1.0], [2.0]], array![[3.0], [4.0]]);
    let _: Dataset<'_, f64, DenseRowMajor<'_>> = Dataset::new(&a).hstack(Dataset::new(&b)).unwrap();
    let _: Dataset<'_, f64, SparseRowMajor<'_>> = Dataset::from_csr(sparse())
        .hstack(Dataset::from_csr(sparse()))
        .unwrap();
}

#[test]
fn stacking_dense_and_sparse_rows_gives_mixed_density_rows() {
    let x = array![[1.0], [2.0]];
    let data: Dataset<'_, f64, Blocks<'_, f64, Rows, MixedDensity>> = Dataset::new(&x)
        .hstack(Dataset::from_csr(sparse()))
        .unwrap();
    // Mixed stays mixed, whatever is added next.
    let _: Dataset<'_, f64, Blocks<'_, f64, Rows, MixedDensity>> =
        data.hstack(Dataset::new(&x)).unwrap();
}

#[test]
fn stacking_column_layouts_keeps_cols_and_mixing_orientations_mixes() {
    let f = Array2::<f64>::zeros((2, 1).f());
    let csc = || CscMatrix::new(2, vec![0, 1], vec![1], vec![3.0]).unwrap();
    let col_major = || Dataset::from_col_major(ColMajorMatrix::new(&f).unwrap());
    let both: Dataset<'_, f64, Blocks<'_, f64, Cols, MixedDensity>> =
        col_major().hstack(Dataset::from_csc(csc())).unwrap();
    assert_eq!(dense(&both), array![[0.0, 0.0], [0.0, 3.0]]);

    let x = array![[1.0], [2.0]];
    let _: Dataset<'_, f64, Blocks<'_, f64, MixedOrientation, Dense>> =
        Dataset::new(&x).hstack(col_major()).unwrap();
    let _: Dataset<'_, f64, Blocks<'_, f64, MixedOrientation, Sparse>> =
        Dataset::from_csr(sparse())
            .hstack(Dataset::from_csc(csc()))
            .unwrap();
}

/// Compiles only if `A + B = C` on that axis.
fn union<A: Union<B, Output = C>, B, C>() {}

#[test]
fn each_axis_keeps_the_same_marker_and_mixes_different_ones() {
    union::<Rows, Rows, Rows>();
    union::<Cols, Cols, Cols>();
    union::<Rows, Cols, MixedOrientation>();
    union::<Cols, MixedOrientation, MixedOrientation>();
    union::<Dense, Dense, Dense>();
    union::<Sparse, Sparse, Sparse>();
    union::<Sparse, Dense, MixedDensity>();
    union::<MixedDensity, MixedDensity, MixedDensity>();
}
