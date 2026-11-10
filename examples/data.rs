//! The feature states, targets, weights and dense and sparse features of a `Dataset`. Run with
//! `cargo run --example data`.

use ndarray::array;
use predict::dataset::{CsrMatrix, Finite};
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
        "{} documents, {} words: {}",
        documents.n_samples(),
        documents.n_features(),
        documents.to_dense()
    );

    // ANCHOR: hstack
    // Each document's length in words: a dense column, beside the sparse counts.
    let lengths = array![[120.0], [80.0], [45.0]];
    let features = Dataset::new(&lengths)
        .validate_finite()?
        .hstack(documents)?;
    // ANCHOR_END: hstack
    println!(
        "{} documents, {} features: {}",
        features.n_samples(),
        features.n_features(),
        features.to_dense()
    );

    Ok(())
}
