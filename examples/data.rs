//! The feature states, targets, weights and dense and sparse features of a `Dataset`. Run with
//! `cargo run --example data`.

use ndarray::ArrayView2;
use ndarray::{Array2, ShapeBuilder, array};
use predict::dataset::{
    BlockView, Blocks, ColMajorMatrix, CscMatrix, CsrMatrix, DenseRowMajor, Finite, MixedDensity,
    Rows, State,
};
use predict::{Dataset, PredictError, Result};

fn main() -> Result<()> {
    // ANCHOR: validate
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    // Borrow the array: nothing is known about the values yet.
    let data = Dataset::new(&x);

    // One scan; the type now records that every value is finite.
    let data: Dataset<'_, _, _, Finite> = data.validate_finite()?;
    // ANCHOR_END: validate
    println!(
        "{} rows, {} features, all finite",
        data.n_samples(),
        data.n_features()
    );

    // ANCHOR: missing
    // NaN marks a missing value. `validate_no_infinity` allows it...
    let with_missing = array![[1.0, f64::NAN], [3.0, 4.0]];
    let data = Dataset::new(&with_missing).validate_no_infinity()?;

    // ...but `validate_finite` reports where it is.
    match data.validate_finite() {
        Err(PredictError::NonFinite { row, col, .. }) => {
            println!("missing value at row {row}, column {col}");
        }
        Err(other) => return Err(other),
        Ok(_) => unreachable!("the data has a NaN"),
    }
    // ANCHOR_END: missing

    // ANCHOR: targets
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // Regression: one float per row, which must be finite.
    let prices = array![10.5, 12.0, 9.75];
    let data = Dataset::new(&x).validate_finite()?.with_y(&prices)?;

    // Classification: one class index per row, 0 to n_classes - 1.
    let classes = array![0_usize, 2, 1];
    let labelled = Dataset::new(&x).with_y(&classes)?;

    // Several outputs: rows × outputs, here two floats per row.
    let both = array![[20.0, 0.5], [21.0, 0.6], [19.5, 0.4]];
    let multi = Dataset::new(&x).with_y(&both)?;
    // ANCHOR_END: targets

    // ANCHOR: weights
    // The second house sold twice: give it weight 2 instead of a second row.
    let counts = array![1.0, 2.0, 1.0];
    let weighted = data.with_sample_weights(&counts)?;
    // ANCHOR_END: weights
    println!(
        "targets: {} prices, {} classes, {:?} outputs; weights {:?}",
        weighted.y().len(),
        labelled.y().len(),
        multi.y().dim(),
        weighted.sample_weights().map(|w| w.to_vec())
    );

    // ANCHOR: sparse
    // How often each of four words appears in each of three documents:
    // [ 5  0  0  1 ]
    // [ 0  0  2  0 ]
    // [ 0  3  0  0 ]
    let row_offsets = vec![0, 2, 3, 4];
    let col_indices = vec![0, 3, 2, 1];
    let values = vec![5.0, 1.0, 2.0, 3.0];
    let counts = CsrMatrix::new(4, row_offsets, col_indices, values)?;
    let documents = Dataset::from_csr(counts).validate_finite()?;
    // ANCHOR_END: sparse
    println!(
        "{} documents, {} words, {} stored",
        documents.n_samples(),
        documents.n_features(),
        match documents.blocks().next() {
            Some(BlockView::Csr(m)) => m.n_stored(),
            _ => unreachable!("from_csr makes one CSR block"),
        }
    );

    // ANCHOR: csc
    // The same word counts, column by column: word 0 appears 5 times in
    // document 0, word 1 3 times in document 2, and so on.
    let col_offsets = vec![0, 1, 2, 3, 4];
    let row_indices = vec![0, 2, 1, 0];
    let values = vec![5.0, 3.0, 2.0, 1.0];
    let by_word = Dataset::from_csc(CscMatrix::new(3, col_offsets, row_indices, values)?);
    // ANCHOR_END: csc
    let counts = array![
        [5.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 0.0]
    ];
    assert_eq!(view(&by_word.into_dense_row_major()), counts);

    // ANCHOR: col_major
    // A column-major (Fortran-order) array, as some file formats and libraries
    // produce: wrapping checks its memory order once, and copies nothing.
    let mut fortran = Array2::zeros((2, 2).f());
    fortran.assign(&array![[1.0, 2.0], [3.0, 4.0]]);
    let by_column = Dataset::from_col_major(ColMajorMatrix::new(&fortran)?);
    assert_eq!(by_column.blocks().len(), 1);
    // ANCHOR_END: col_major

    // ANCHOR: hstack
    // Each document's length in words: a dense column, beside the sparse counts.
    let lengths = array![[120.0], [80.0], [45.0]];
    // Both sides run by rows; one is dense and one sparse.
    let features: Dataset<'_, f64, Blocks<'_, f64, Rows, MixedDensity>, Finite> =
        Dataset::new(&lengths)
            .validate_finite()?
            .hstack(documents)?;
    // ANCHOR_END: hstack
    println!(
        "{} documents, {} features in {} blocks",
        features.n_samples(),
        features.n_features(),
        features.blocks().len()
    );

    // ANCHOR: convert
    // A solver that needs one row-major matrix: the conversion is written out,
    // and it is the only place the features are copied.
    let dense: Dataset<'_, f64, DenseRowMajor<'_>, Finite> = features.into_dense_row_major();
    // ANCHOR_END: convert
    println!("as one matrix: {}", view(&dense));

    Ok(())
}

/// The one dense block of a dataset converted to `DenseRowMajor`, borrowed.
fn view<'b, 'a, S: State>(data: &'b Dataset<'a, f64, DenseRowMajor<'a>, S>) -> ArrayView2<'b, f64> {
    match data.blocks().next() {
        Some(BlockView::DenseRowMajor(x)) => x,
        _ => unreachable!("a conversion makes one block"),
    }
}
