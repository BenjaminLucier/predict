# predict

Machine learning for Rust with a consistent, scikit-learn-style API.

[Book](https://predict.rs) · [API docs](https://docs.rs/predict) ·
[Contributing](CONTRIBUTING.md)

> **Status:** early development, `0.0.x`. Every release may break the API.

**Polars to production.** predict aims to cover the whole path from a
dataframe to a model running in production, in Rust, for the algorithms
practitioners actually reach for.

## What predict aims for

- **Speed:** traditional ML as fast as a single machine allows.
- **Correctness:** results checked against scikit-learn, every input
  validated, and errors instead of silent NaN.
- **Type safety:** a dataset's checks are part of its type, so fitting
  unvalidated data, or sparse data to a dense-only model, doesn't compile.
- **Rust-native:** pure Rust, with no Python runtime and no C or BLAS
  libraries to install.
- **Batteries included:** the pieces around a model ship with it, from
  splitting and cross-validation to metrics.

It is selective on purpose. An algorithm is included because people use it
in practice, not because it exists in the literature, and each one is done
properly rather than added for the count.

## What's here

- **Datasets** that carry their checks in the type: dense (row- or
  column-major) and sparse (CSR, CSC) features, side by side, with targets and
  sample weights, validated once and borrowed rather than copied.
- **Synthetic data:** regression, blobs and classification generators, as
  whole datasets or endless streams.
- **Metrics:** MSE, RMSE, MAE and R²; accuracy, precision, recall and F-scores,
  all with sample weights.
- **Model selection:** train/test splits, k-fold and stratified k-fold, and
  cross-validation.
- **Linear regression:** ordinary least squares, solved by SVD, matching
  scikit-learn's results.

## Example

```rust
use predict::linear::LinearRegression;
use predict::metrics::R2;
use predict::selection::{CrossValidation, KFold};
use predict::synthetic::MakeRegression;
use predict::{Fit, Result, Seed};

fn main() -> Result<()> {
    let generated = MakeRegression::new()
        .with_n_samples(200)
        .with_n_features(3)
        .with_n_informative(3)
        .with_noise(0.5)
        .with_seed(0)
        .generate()?;
    let data = generated.dataset()?;

    let model = LinearRegression::new().fit(&data)?;
    println!("fitted coefficients: {:.3}", model.coef());

    let cv = CrossValidation::new(KFold::new().with_shuffle(Seed::Fixed(42)));
    let scores = cv.score(&LinearRegression::new(), &data, (), &R2)?;
    println!("R² per fold: {scores:.4}");
    Ok(())
}
```

The same code is in [`examples/linear.rs`](examples/linear.rs):
`cargo run --example linear`.

## Minimum Rust version

Rust 1.88. Raising it is a minor-version change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
