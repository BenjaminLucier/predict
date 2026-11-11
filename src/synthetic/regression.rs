//! Linear regression data: [`MakeRegression`].

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use rand::RngExt;
use rand::distr::OpenClosed01;
use rand::rngs::Xoshiro256PlusPlus;
use rand_distr::StandardNormal;

use super::Samples;
use crate::dataset::{DenseRowMajor, Finite, Targets};
use crate::error::require;
use crate::{Dataset, Result, Seed};

/// Gaussian features and a target that is a linear function of some of them.
///
/// Every feature is drawn from `N(0, 1)`. The first `n_informative` features
/// get a true coefficient drawn uniformly from `(0, 100]`, the rest get 0, and
///
/// ```text
/// y = x · coef + bias + noise · N(0, 1)
/// ```
///
/// A linear model fitted on enough rows should recover `coef` and `bias`, to
/// within about `noise / sqrt(n_samples)` per coefficient, and exactly (up to
/// rounding) when `noise` is 0. The derivation is in the book:
/// <https://predict.rs/synthetic.html>.
///
/// The rows of [`generate`](Self::generate) are the first `n_samples` rows of
/// [`stream`](Self::stream) with the same seed, however they are batched.
#[derive(Debug, Clone)]
pub struct MakeRegression {
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    bias: f64,
    seed: Seed,
}

impl Default for MakeRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl MakeRegression {
    /// 100 samples of 10 features, 5 of them informative, no noise, no bias,
    /// and a different seed every run.
    pub fn new() -> Self {
        Self {
            n_samples: 100,
            n_features: 10,
            n_informative: 5,
            noise: 0.0,
            bias: 0.0,
            seed: Seed::default(),
        }
    }

    /// The number of rows `generate` draws. Streams ignore it.
    pub fn with_n_samples(self, n_samples: usize) -> Self {
        Self { n_samples, ..self }
    }

    /// The number of features.
    pub fn with_n_features(self, n_features: usize) -> Self {
        Self { n_features, ..self }
    }

    /// The number of features with a nonzero coefficient: the first ones.
    pub fn with_n_informative(self, n_informative: usize) -> Self {
        Self {
            n_informative,
            ..self
        }
    }

    /// The standard deviation of the Gaussian noise added to each target.
    pub fn with_noise(self, noise: f64) -> Self {
        Self { noise, ..self }
    }

    /// The constant added to every target: the true intercept.
    pub fn with_bias(self, bias: f64) -> Self {
        Self { bias, ..self }
    }

    /// Draw from a fixed seed, so every run gives the same data.
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Seed::Fixed(seed),
            ..self
        }
    }

    /// Draw `n_samples` rows, with the coefficients and intercept behind them.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` if `n_samples` or `n_features` is 0, `n_informative`
    /// is greater than `n_features`, `noise` is negative or not finite, or
    /// `bias` is not finite.
    pub fn generate(&self) -> Result<RegressionData> {
        require(self.n_samples > 0, "n_samples", "must be positive")?;
        let mut stream = self.stream()?;
        let Samples { x, y } = stream.next_batch(self.n_samples);
        Ok(RegressionData {
            x,
            y,
            coef: stream.coef,
            intercept: self.bias,
        })
    }

    /// An endless stream of rows: the coefficients are drawn now, then each
    /// row as it is asked for.
    ///
    /// # Errors
    ///
    /// As for [`generate`](Self::generate), except that `n_samples` is not
    /// used.
    pub fn stream(&self) -> Result<RegressionStream> {
        require(self.n_features > 0, "n_features", "must be positive")?;
        require(
            self.n_informative <= self.n_features,
            "n_informative",
            "must not exceed n_features",
        )?;
        require(
            self.noise.is_finite() && self.noise >= 0.0,
            "noise",
            "must be finite and non-negative",
        )?;
        require(self.bias.is_finite(), "bias", "must be finite")?;

        let mut rng = self.seed.rng();
        let coef = Array1::from_shape_fn(self.n_features, |feature| {
            if feature < self.n_informative {
                100.0 * rng.sample::<f64, _>(OpenClosed01)
            } else {
                0.0
            }
        });
        Ok(RegressionStream {
            coef,
            bias: self.bias,
            noise: self.noise,
            rng,
        })
    }
}

/// What [`MakeRegression::generate`] returns: the rows and the truth.
#[derive(Debug, Clone)]
pub struct RegressionData {
    /// The features, `n_samples × n_features`.
    pub x: Array2<f64>,
    /// The targets: `x · coef + intercept + noise`.
    pub y: Array1<f64>,
    /// The true coefficients, 0 on the features that aren't informative.
    pub coef: Array1<f64>,
    /// The true intercept, the `bias` setting.
    pub intercept: f64,
}

impl RegressionData {
    /// `x` and `y` as a dataset, borrowing them: validated as finite, with `y`
    /// as its targets.
    ///
    /// # Errors
    ///
    /// `NonFinite` if a value overflowed, which only extreme settings cause.
    pub fn dataset(&self) -> Result<Dataset<'_, f64, DenseRowMajor<'_>, Finite, Targets<'_, f64>>> {
        super::dataset(&self.x, &self.y)
    }
}

/// An endless stream of regression rows, from [`MakeRegression::stream`].
///
/// As an iterator it yields one `(x, y)` row at a time and never ends; use
/// `take(n)` to stop. [`next_batch`](Self::next_batch) and
/// [`batches`](Self::batches) draw several rows into one matrix. The rows are
/// the same however they are drawn: a seed fixes the sequence, and batching
/// only groups it.
#[derive(Debug, Clone)]
pub struct RegressionStream {
    coef: Array1<f64>,
    bias: f64,
    noise: f64,
    rng: Xoshiro256PlusPlus,
}

impl RegressionStream {
    /// The true coefficients.
    pub fn coef(&self) -> ArrayView1<'_, f64> {
        self.coef.view()
    }

    /// The true intercept.
    pub fn intercept(&self) -> f64 {
        self.bias
    }

    /// The next `n` rows, as one matrix and their targets.
    pub fn next_batch(&mut self, n: usize) -> Samples<f64> {
        let mut x = Array2::zeros((n, self.coef.len()));
        let y = x
            .rows_mut()
            .into_iter()
            .map(|row| self.fill_row(row))
            .collect();
        Samples { x, y }
    }

    /// An endless iterator of batches of `batch_size` rows each.
    ///
    /// # Panics
    ///
    /// If `batch_size` is 0, which would never draw a row (as with
    /// `slice::chunks`).
    pub fn batches(&mut self, batch_size: usize) -> impl Iterator<Item = Samples<f64>> + '_ {
        assert!(batch_size > 0, "batch_size must be positive");
        std::iter::repeat_with(move || self.next_batch(batch_size))
    }

    /// Draw one row into `x` and return its target.
    ///
    /// The noise is drawn even when `noise` is 0, so a seed gives the same
    /// features at every noise level and only `y` changes.
    fn fill_row(&mut self, mut x: ArrayViewMut1<'_, f64>) -> f64 {
        x.mapv_inplace(|_| self.rng.sample(StandardNormal));
        let epsilon: f64 = self.rng.sample(StandardNormal);
        x.dot(&self.coef) + self.bias + self.noise * epsilon
    }
}

impl Iterator for RegressionStream {
    type Item = (Array1<f64>, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let mut x = Array1::zeros(self.coef.len());
        let y = self.fill_row(x.view_mut());
        Some((x, y))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}
