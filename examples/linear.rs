//! Ordinary least squares: recover known coefficients, then cross-validate.
//! Run with `cargo run --example linear`.

use predict::linear::LinearRegression;
use predict::metrics::R2;
use predict::selection::{CrossValidation, KFold};
use predict::synthetic::MakeRegression;
use predict::{Fit, Result, Seed};

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
    Ok(())
}
