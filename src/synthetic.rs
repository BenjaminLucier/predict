//! Synthetic data with a known truth, for checking that a model recovers it.
//! See <https://predict.rs/synthetic.html>.
//!
//! Each generator is a config, like an estimator: `new()`, then `with_*`
//! settings, then one of two ways to draw:
//!
//! - `generate()` draws `n_samples` rows and returns them with the truth they
//!   were drawn from (true coefficients, true centres).
//! - `stream()` returns an endless iterator of rows, with the same truth
//!   readable on it. `next_batch(n)` and `batches(n)` draw several rows at a
//!   time.
//!
//! Generators draw from a [`Seed`](crate::Seed), set with `with_seed(42)`; by
//! default every run differs.

mod blobs;
mod classification;
mod regression;

pub use blobs::{BlobsData, BlobsStream, MakeBlobs};
pub use classification::{ClassificationData, ClassificationStream, MakeClassification};
pub use regression::{MakeRegression, RegressionData, RegressionStream};

use std::iter;

use ndarray::{Array1, Array2};
use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::SliceRandom;

use crate::Dataset;
use crate::Result;
use crate::dataset::{DenseRowMajor, Finite, Target, Targets};

/// Rows drawn from a stream: features and their targets.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples<T> {
    /// The features, rows × features.
    pub x: Array2<f64>,
    /// The targets, one per row.
    pub y: Array1<T>,
}

impl<T: Target> Samples<T> {
    /// The rows as a dataset, borrowing `x` and `y`: validated as finite, with
    /// `y` as its targets.
    ///
    /// # Errors
    ///
    /// `NonFinite` if a value overflowed, which only extreme settings cause.
    pub fn dataset(&self) -> Result<Dataset<'_, f64, DenseRowMajor<'_>, Finite, Targets<'_, T>>> {
        dataset(&self.x, &self.y)
    }
}

/// `x` and `y` as a finite dataset with targets, scanned once by the public
/// validation.
fn dataset<'a, T: Target>(
    x: &'a Array2<f64>,
    y: &'a Array1<T>,
) -> Result<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, T>>> {
    Dataset::new(x).validate_finite()?.with_y(y)
}

/// One class per row, in random order, with class sizes in proportion to
/// `weights`: what `generate` uses, so a small dataset still gets the requested
/// balance. Streams draw each row's class independently instead.
fn shuffled_classes(n_samples: usize, weights: &[f64], rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
    let mut classes: Vec<usize> = class_sizes(n_samples, weights)
        .into_iter()
        .enumerate()
        .flat_map(|(class, size)| iter::repeat_n(class, size))
        .collect();
    classes.shuffle(rng);
    classes
}

/// Largest-remainder rounding of each class's share `n_samples * weight /
/// total`: the sizes add up to `n_samples`, each is within one of its share,
/// and ties for the leftover rows go to the lower class.
fn class_sizes(n_samples: usize, weights: &[f64]) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    // Divide first, so huge finite weights can't overflow.
    let shares: Vec<f64> = weights
        .iter()
        .map(|weight| weight / total * n_samples as f64)
        .collect();
    let mut sizes: Vec<usize> = shares.iter().map(|share| share.floor() as usize).collect();
    let leftover = n_samples - sizes.iter().sum::<usize>();
    let mut by_remainder: Vec<usize> = (0..weights.len()).collect();
    // A stable sort keeps lower classes first among equal remainders.
    by_remainder.sort_by(|&a, &b| shares[b].fract().total_cmp(&shares[a].fract()));
    for &class in by_remainder.iter().take(leftover) {
        sizes[class] += 1;
    }
    sizes
}
