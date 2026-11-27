//! Ordinary least squares: recover known coefficients, then cross-validate.
//! Ridge regression: stable coefficients on nearly collinear columns.
//! Run with `cargo run --example linear`.

use ndarray::{Array1, Array2};
use predict::linear::{LinearRegression, RidgeRegression};
use predict::metrics::R2;
use predict::selection::{CrossValidation, KFold};
use predict::synthetic::MakeRegression;
use predict::{Dataset, Fit, Result, Seed};

fn main() -> Result<()> {
    // ANCHOR: fit
    let generated = MakeRegression::new()
        .with_n_samples(200)
        .with_n_features(3)
        .with_n_informative(3)
        .with_bias(2.0)
        .with_noise(0.5)
        .with_seed(0)
        .generate()?;
    let data = generated.dataset()?;

    let model = LinearRegression::new().fit(&data)?;
    println!("true coefficients:   {:.3}", generated.coef);
    println!("fitted coefficients: {:.3}", model.coef());
    println!(
        "true intercept {:.3}, fitted {:.3}",
        generated.intercept,
        model.intercept()
    );
    // ANCHOR_END: fit

    // ANCHOR: cross_validation
    let cv = CrossValidation::new(KFold::new().with_shuffle(Seed::Fixed(42)));
    let scores = cv.score(&LinearRegression::new(), &data, (), &R2)?;
    println!("R² per fold: {scores:.4}");
    // ANCHOR_END: cross_validation

    // ANCHOR: ridge
    // Two columns that are almost the same measurement, and a target that is
    // their sum plus noise.
    let n = 50;
    let first = Array1::from_shape_fn(n, |i| i as f64 / 10.0);
    let second = Array1::from_shape_fn(n, |i| i as f64 / 10.0 + 0.001 * (7.3 * i as f64).sin());
    let noise = Array1::from_shape_fn(n, |i| 0.5 * (1.7 * i as f64).cos());
    let y = &first + &second + noise;
    let mut x = Array2::zeros((n, 2));
    x.column_mut(0).assign(&first);
    x.column_mut(1).assign(&second);
    let data = Dataset::new(&x).validate_finite()?.with_y(&y)?;

    let ols = LinearRegression::new().fit(&data)?;
    let ridge = RidgeRegression::new().with_alpha(1.0).fit(&data)?;
    println!("least squares coefficients: {:.3}", ols.coef());
    println!("ridge coefficients:         {:.3}", ridge.coef());
    // ANCHOR_END: ridge
    Ok(())
}
