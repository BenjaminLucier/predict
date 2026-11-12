//! Train/test splits and cross-validation, with a baseline that always
//! predicts the training mean. Run with `cargo run --example selection`.

use ndarray::Array1;
use predict::dataset::{DenseRowMajor, Finite, TargetState, Targets};
use predict::metrics::R2;
use predict::selection::{CrossValidation, KFold, Splitter, TrainTest, TrainTestSplit};
use predict::synthetic::MakeRegression;
use predict::{Dataset, Fit, Predict, Result, Seed, ValidateConfig};

// ANCHOR: baseline
/// Predicts the mean of the training targets for every row: the score any
/// real model has to beat (scikit-learn's `DummyRegressor`).
#[derive(Debug, Default)]
struct MeanBaseline;

struct FittedMeanBaseline(f64);

impl ValidateConfig for MeanBaseline {}

impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>> for MeanBaseline {
    type Fitted = FittedMeanBaseline;

    fn fit(
        &self,
        data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>,
    ) -> Result<FittedMeanBaseline> {
        Ok(FittedMeanBaseline(data.y().mean().unwrap_or(0.0)))
    }
}

impl<'a, Y: TargetState> Predict<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>>
    for FittedMeanBaseline
{
    type Output = Array1<f64>;

    fn predict(&self, x: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(x.n_samples(), self.0))
    }
}
// ANCHOR_END: baseline

fn main() -> Result<()> {
    let generated = MakeRegression::new()
        .with_n_samples(200)
        .with_noise(1.0)
        .with_seed(0)
        .generate()?;
    let data = generated.dataset()?;

    // ANCHOR: split
    let split = TrainTestSplit::new()
        .with_test_size(0.2)
        .with_seed(42)
        .splits(data.n_samples(), ())?
        .remove(0);
    let TrainTest { train, test } = split.select(&data);
    // ANCHOR_END: split
    println!(
        "{} training rows, {} test rows",
        train.n_samples(),
        test.n_samples()
    );

    // ANCHOR: cross_validation
    let cv = CrossValidation::new(KFold::new().with_shuffle(Seed::Fixed(42)));
    let scores = cv.score(&MeanBaseline, &data, (), &R2)?;
    println!(
        "R² per fold: {scores:.3}, mean {:.3} ± {:.3}",
        scores.mean().unwrap(),
        scores.std(1.0)
    );
    // ANCHOR_END: cross_validation
    Ok(())
}
