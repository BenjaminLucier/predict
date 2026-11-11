//! Clusters of points around random centres: [`MakeBlobs`].

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut1};
use rand::RngExt;
use rand::distr::Uniform;
use rand::rngs::Xoshiro256PlusPlus;
use rand_distr::StandardNormal;

use super::{Samples, shuffled_classes};
use crate::dataset::{DenseRowMajor, Finite, Targets};
use crate::error::require;
use crate::{Dataset, PredictError, Result, Seed};

/// Round Gaussian clusters around random centres: the standard clustering
/// problem.
///
/// Each centre's coordinates are drawn uniformly from the centre box, and each
/// row is its centre plus `N(0, cluster_std²)` noise on every feature. The
/// target is the centre a row was drawn around. When the clusters are far
/// apart compared with `cluster_std`, k-means should find centres close to the
/// true ones; as `cluster_std` grows they overlap and no method can separate
/// them.
///
/// [`generate`](Self::generate) splits the rows evenly between the centres
/// (within one row) and shuffles them. A [`stream`](Self::stream) picks each
/// row's centre at random, so its first rows are not those of `generate`.
#[derive(Debug, Clone)]
pub struct MakeBlobs {
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    center_box: (f64, f64),
    seed: Seed,
}

impl Default for MakeBlobs {
    fn default() -> Self {
        Self::new()
    }
}

impl MakeBlobs {
    /// 100 samples of 2 features around 3 centres, `cluster_std` 1, centres in
    /// `(-10, 10)`, and a different seed every run.
    pub fn new() -> Self {
        Self {
            n_samples: 100,
            n_features: 2,
            n_centers: 3,
            cluster_std: 1.0,
            center_box: (-10.0, 10.0),
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

    /// The number of clusters.
    pub fn with_n_centers(self, n_centers: usize) -> Self {
        Self { n_centers, ..self }
    }

    /// The standard deviation of every cluster, on every feature.
    pub fn with_cluster_std(self, cluster_std: f64) -> Self {
        Self {
            cluster_std,
            ..self
        }
    }

    /// The range each centre coordinate is drawn from, `low..high`.
    pub fn with_center_box(self, low: f64, high: f64) -> Self {
        Self {
            center_box: (low, high),
            ..self
        }
    }

    /// Draw from a fixed seed, so every run gives the same data.
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Seed::Fixed(seed),
            ..self
        }
    }

    /// Draw `n_samples` rows, split evenly between the centres, with the
    /// centres.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` if `n_samples`, `n_features` or `n_centers` is 0,
    /// `cluster_std` is negative or not finite, or the centre box isn't finite
    /// with `low < high`.
    pub fn generate(&self) -> Result<BlobsData> {
        require(self.n_samples > 0, "n_samples", "must be positive")?;
        let mut stream = self.stream()?;
        let classes = shuffled_classes(self.n_samples, &vec![1.0; self.n_centers], &mut stream.rng);
        let mut x = Array2::zeros((self.n_samples, self.n_features));
        for (row, &class) in x.rows_mut().into_iter().zip(&classes) {
            stream.fill_row(class, row);
        }
        Ok(BlobsData {
            x,
            y: Array1::from(classes),
            centers: stream.centers,
        })
    }

    /// An endless stream of rows: the centres are drawn now, then each row,
    /// around a centre picked at random, as it is asked for.
    ///
    /// # Errors
    ///
    /// As for [`generate`](Self::generate), except that `n_samples` is not
    /// used.
    pub fn stream(&self) -> Result<BlobsStream> {
        require(self.n_features > 0, "n_features", "must be positive")?;
        require(self.n_centers > 0, "n_centers", "must be positive")?;
        require(
            self.cluster_std.is_finite() && self.cluster_std >= 0.0,
            "cluster_std",
            "must be finite and non-negative",
        )?;
        let (low, high) = self.center_box;
        // `Uniform` rejects a box that isn't finite, is empty, or whose width
        // overflows.
        let coordinate = Uniform::new(low, high).map_err(|_| PredictError::InvalidParameter {
            name: "center_box",
            reason: "must be finite with low < high".to_string(),
        })?;

        let mut rng = self.seed.rng();
        let centers = Array2::from_shape_simple_fn((self.n_centers, self.n_features), || {
            rng.sample(coordinate)
        });
        Ok(BlobsStream {
            centers,
            cluster_std: self.cluster_std,
            rng,
        })
    }
}

/// What [`MakeBlobs::generate`] returns: the rows and the true centres.
#[derive(Debug, Clone)]
pub struct BlobsData {
    /// The features, `n_samples × n_features`.
    pub x: Array2<f64>,
    /// The centre each row was drawn around, in `0..n_centers`.
    pub y: Array1<usize>,
    /// The true centres, `n_centers × n_features`: row `c` is centre `c`.
    pub centers: Array2<f64>,
}

impl BlobsData {
    /// `x` and `y` as a dataset, borrowing them: validated as finite, with `y`
    /// as its targets.
    ///
    /// # Errors
    ///
    /// `NonFinite` if a value overflowed, which only extreme settings cause.
    pub fn dataset(
        &self,
    ) -> Result<Dataset<'_, f64, DenseRowMajor<'_>, Finite, Targets<'_, usize>>> {
        super::dataset(&self.x, &self.y)
    }
}

/// An endless stream of rows around random centres, from
/// [`MakeBlobs::stream`].
///
/// As an iterator it yields one `(x, centre)` row at a time and never ends;
/// [`next_batch`](Self::next_batch) and [`batches`](Self::batches) draw several
/// rows into one matrix, the same rows however they are grouped.
#[derive(Debug, Clone)]
pub struct BlobsStream {
    centers: Array2<f64>,
    cluster_std: f64,
    rng: Xoshiro256PlusPlus,
}

impl BlobsStream {
    /// The true centres, one per row.
    pub fn centers(&self) -> ArrayView2<'_, f64> {
        self.centers.view()
    }

    /// The next `n` rows, as one matrix and their centres.
    pub fn next_batch(&mut self, n: usize) -> Samples<usize> {
        let mut x = Array2::zeros((n, self.centers.ncols()));
        let y = x
            .rows_mut()
            .into_iter()
            .map(|row| self.next_row(row))
            .collect();
        Samples { x, y }
    }

    /// An endless iterator of batches of `batch_size` rows each.
    ///
    /// # Panics
    ///
    /// If `batch_size` is 0, which would never draw a row (as with
    /// `slice::chunks`).
    pub fn batches(&mut self, batch_size: usize) -> impl Iterator<Item = Samples<usize>> + '_ {
        assert!(batch_size > 0, "batch_size must be positive");
        std::iter::repeat_with(move || self.next_batch(batch_size))
    }

    /// Pick a centre at random, draw a row around it into `x`, and return the
    /// centre.
    fn next_row(&mut self, x: ArrayViewMut1<'_, f64>) -> usize {
        let class = self.rng.random_range(0..self.centers.nrows());
        self.fill_row(class, x);
        class
    }

    /// Draw a row around centre `class` into `x`.
    fn fill_row(&mut self, class: usize, mut x: ArrayViewMut1<'_, f64>) {
        for (value, &center) in x.iter_mut().zip(self.centers.row(class)) {
            let z: f64 = self.rng.sample(StandardNormal);
            *value = center + self.cluster_std * z;
        }
    }
}

impl Iterator for BlobsStream {
    type Item = (Array1<f64>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let mut x = Array1::zeros(self.centers.ncols());
        let class = self.next_row(x.view_mut());
        Some((x, class))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}
