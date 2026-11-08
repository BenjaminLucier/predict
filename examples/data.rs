//! The feature states of a `Dataset`. Run with `cargo run --example data`.

use ndarray::array;
use predict::dataset::{Finite, NoInfinity};
use predict::{Dataset, PredictError, Result};

fn main() -> Result<()> {
    // ANCHOR: validate
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    // Borrow the array: nothing is known about the values yet.
    let data = Dataset::new(&x);

    // One scan; the type now records that every value is finite.
    let data: Dataset<'_, _, Finite> = data.validate_finite()?;
    // ANCHOR_END: validate
    println!(
        "{} rows, {} features, all finite",
        data.n_samples(),
        data.n_features()
    );

    // ANCHOR: missing
    // NaN marks a missing value. `validate_no_infinity` allows it...
    let with_missing = array![[1.0, f64::NAN], [3.0, 4.0]];
    let data: Dataset<'_, _, NoInfinity> = Dataset::new(&with_missing).validate_no_infinity()?;

    // ...but `validate_finite` reports where it is.
    match data.validate_finite() {
        Err(PredictError::NonFinite { row, col, .. }) => {
            println!("missing value at row {row}, column {col}");
        }
        Err(other) => return Err(other),
        Ok(_) => unreachable!("the data has a NaN"),
    }
    // ANCHOR_END: missing

    Ok(())
}
