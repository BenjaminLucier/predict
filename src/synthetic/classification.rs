//! Classes around the corners of a cube: [`MakeClassification`].

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut1, s};
use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::index;
use rand_distr::StandardNormal;

use super::{Samples, shuffled_classes};
use crate::dataset::{DenseRowMajor, Finite, Targets};
use crate::error::require;
use crate::{Dataset, Result, Seed};

/// A classification problem whose difficulty is set by how far apart the
/// classes are, with informative, redundant, repeated and useless features.
///
/// Each class gets its own random corner of a cube in the first
/// `n_informative` dimensions, every coordinate `±class_sep`, and each row is
/// its class's corner plus `N(0, 1)` noise. Two classes whose corners differ on
/// `m` axes are `2 · class_sep · sqrt(m)` apart, so even the best possible
/// classifier mistakes one for the other with probability `Φ(-class_sep ·
/// sqrt(m))`; the book derives it at <https://predict.rs/synthetic.html>.
///
/// The columns, in order:
///
/// 1. `n_informative` informative features: the corner plus noise.
/// 2. `n_redundant` features, each a random linear combination of the
///    informative ones (weights uniform in `[-1, 1)`), so correlated with them.
/// 3. `n_repeated` exact copies of random informative or redundant features.
/// 4. The rest: `N(0, 1)` noise, unrelated to the class.
///
/// Last, each row's class is replaced by a random class (possibly the same)
/// with probability `flip_y`, so that not even the best classifier can be
/// perfect, as in real data.
///
/// [`generate`](Self::generate) splits the rows between the classes in
/// proportion to `class_weights` (within one row, before flipping) and
/// shuffles them. A [`stream`](Self::stream) picks each row's class at random
/// with those weights, so its first rows are not those of `generate`.
///
/// This is scikit-learn's `make_classification` with one cluster per class, no
/// random covariance, and no column shuffling.
#[derive(Debug, Clone)]
pub struct MakeClassification {
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_redundant: usize,
    n_repeated: usize,
    n_classes: usize,
    class_sep: f64,
    class_weights: Option<Vec<f64>>,
    flip_y: f64,
    seed: Seed,
}

impl Default for MakeClassification {
    fn default() -> Self {
        Self::new()
    }
}

impl MakeClassification {
    /// As scikit-learn: 100 samples of 20 features (2 informative, 2
    /// redundant, no repeated), 2 balanced classes, `class_sep` 1, `flip_y`
    /// 0.01, and a different seed every run.
    pub fn new() -> Self {
        Self {
            n_samples: 100,
            n_features: 20,
            n_informative: 2,
            n_redundant: 2,
            n_repeated: 0,
            n_classes: 2,
            class_sep: 1.0,
            class_weights: None,
            flip_y: 0.01,
            seed: Seed::default(),
        }
    }

    /// The number of rows `generate` draws. Streams ignore it.
    pub fn with_n_samples(self, n_samples: usize) -> Self {
        Self { n_samples, ..self }
    }

    /// The number of features of every kind.
    pub fn with_n_features(self, n_features: usize) -> Self {
        Self { n_features, ..self }
    }

    /// The number of features that place a row at its class's corner.
    pub fn with_n_informative(self, n_informative: usize) -> Self {
        Self {
            n_informative,
            ..self
        }
    }

    /// The number of features that are linear combinations of the informative
    /// ones.
    pub fn with_n_redundant(self, n_redundant: usize) -> Self {
        Self {
            n_redundant,
            ..self
        }
    }

    /// The number of features that copy an informative or redundant one.
    pub fn with_n_repeated(self, n_repeated: usize) -> Self {
        Self { n_repeated, ..self }
    }

    /// The number of classes.
    pub fn with_n_classes(self, n_classes: usize) -> Self {
        Self { n_classes, ..self }
    }

    /// How far each corner coordinate is from 0: larger is easier.
    pub fn with_class_sep(self, class_sep: f64) -> Self {
        Self { class_sep, ..self }
    }

    /// The relative size of each class, one weight per class. They needn't
    /// add up to 1. Balanced by default.
    pub fn with_class_weights(self, class_weights: Vec<f64>) -> Self {
        Self {
            class_weights: Some(class_weights),
            ..self
        }
    }

    /// The probability that a row's class is replaced by a random one.
    pub fn with_flip_y(self, flip_y: f64) -> Self {
        Self { flip_y, ..self }
    }

    /// Draw from a fixed seed, so every run gives the same data.
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Seed::Fixed(seed),
            ..self
        }
    }

    /// Draw `n_samples` rows, the classes in proportion to the weights, with
    /// the truth behind them.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` if `n_samples`, `n_classes` or `n_informative` is 0;
    /// the informative, redundant and repeated features add up to more than
    /// `n_features`; `n_classes` is more than the cube's `2^n_informative`
    /// corners; `class_sep` isn't finite and positive; `flip_y` isn't in
    /// `[0, 1]`; or `class_weights` doesn't have one finite, non-negative
    /// weight per class with a finite, positive total.
    pub fn generate(&self) -> Result<ClassificationData> {
        require(self.n_samples > 0, "n_samples", "must be positive")?;
        let mut stream = self.stream()?;
        let classes = shuffled_classes(self.n_samples, &self.weights(), &mut stream.rng);
        let mut x = Array2::zeros((self.n_samples, self.n_features));
        let y = x
            .rows_mut()
            .into_iter()
            .zip(classes)
            .map(|(row, class)| stream.fill_row(class, row))
            .collect();
        Ok(ClassificationData {
            x,
            y,
            centers: stream.centers,
            redundant: stream.redundant,
            repeated: stream.repeated,
        })
    }

    /// An endless stream of rows: the corners and feature mixes are drawn now,
    /// then each row, its class picked at random with the class weights, as it
    /// is asked for.
    ///
    /// # Errors
    ///
    /// As for [`generate`](Self::generate), except that `n_samples` is not
    /// used.
    pub fn stream(&self) -> Result<ClassificationStream> {
        self.check()?;
        let mut rng = self.seed.rng();
        let centers = corners(self.n_classes, self.n_informative, self.class_sep, &mut rng);
        let redundant =
            Array2::from_shape_simple_fn((self.n_informative, self.n_redundant), || {
                rng.random_range(-1.0..1.0)
            });
        let n_copyable = self.n_informative + self.n_redundant;
        let repeated = (0..self.n_repeated)
            .map(|_| rng.random_range(0..n_copyable))
            .collect();
        let class = WeightedIndex::new(self.weights()).expect("weights checked above");
        Ok(ClassificationStream {
            centers,
            redundant,
            repeated,
            n_features: self.n_features,
            flip_y: self.flip_y,
            class,
            rng,
        })
    }

    fn weights(&self) -> Vec<f64> {
        self.class_weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.n_classes])
    }

    fn check(&self) -> Result<()> {
        require(self.n_classes > 0, "n_classes", "must be positive")?;
        require(self.n_informative > 0, "n_informative", "must be positive")?;
        let n_used = self
            .n_informative
            .checked_add(self.n_redundant)
            .and_then(|n| n.checked_add(self.n_repeated));
        require(
            n_used.is_some_and(|n| n <= self.n_features),
            "n_features",
            "must be at least n_informative + n_redundant + n_repeated",
        )?;
        // The corners are drawn as distinct integers below 2^63 at most, so
        // that bounds the classes even with more dimensions.
        let n_corners = 1_usize << self.n_informative.min(usize::BITS as usize - 1);
        require(
            self.n_classes <= n_corners,
            "n_classes",
            "must not exceed 2^n_informative, the number of corners",
        )?;
        require(
            self.class_sep.is_finite() && self.class_sep > 0.0,
            "class_sep",
            "must be finite and positive",
        )?;
        require(
            (0.0..=1.0).contains(&self.flip_y),
            "flip_y",
            "must be between 0 and 1",
        )?;
        if let Some(weights) = &self.class_weights {
            require(
                weights.len() == self.n_classes,
                "class_weights",
                "must have one weight per class",
            )?;
            require(
                weights.iter().all(|w| w.is_finite() && *w >= 0.0),
                "class_weights",
                "must be finite and non-negative",
            )?;
            let total: f64 = weights.iter().sum();
            require(
                total.is_finite() && total > 0.0,
                "class_weights",
                "must have a finite, positive total",
            )?;
        }
        Ok(())
    }
}

/// `n_corners` distinct random corners of the cube with coordinates
/// `±side` in `n_dims` dimensions, one per row.
///
/// The bits of an integer pick a corner's signs, so distinct integers are
/// distinct corners, and drawing them without replacement takes the same time
/// however many of the corners are used. An integer covers at most
/// `usize::BITS - 1` dimensions; any further ones get independent random signs.
fn corners(
    n_corners: usize,
    n_dims: usize,
    side: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Array2<f64> {
    let n_bits = n_dims.min(usize::BITS as usize - 1);
    let ids = index::sample(rng, 1 << n_bits, n_corners);
    let mut corners = Array2::zeros((n_corners, n_dims));
    for (mut corner, id) in corners.rows_mut().into_iter().zip(ids.iter()) {
        for (dim, value) in corner.iter_mut().enumerate() {
            let positive = if dim < n_bits {
                (id >> dim) & 1 == 1
            } else {
                rng.random()
            };
            *value = if positive { side } else { -side };
        }
    }
    corners
}

/// What [`MakeClassification::generate`] returns: the rows and the truth.
#[derive(Debug, Clone)]
pub struct ClassificationData {
    /// The features, `n_samples × n_features`, in the order informative,
    /// redundant, repeated, useless.
    pub x: Array2<f64>,
    /// The class of each row, in `0..n_classes`, after flipping.
    pub y: Array1<usize>,
    /// The class corners, `n_classes × n_informative`: row `c` is the centre
    /// of class `c` in the informative features.
    pub centers: Array2<f64>,
    /// The mix behind the redundant features, `n_informative × n_redundant`:
    /// redundant feature `r` is the informative features times column `r`.
    pub redundant: Array2<f64>,
    /// Which feature each repeated feature copies, one index per repeated
    /// feature, counting from the first informative feature.
    pub repeated: Vec<usize>,
}

impl ClassificationData {
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

/// An endless stream of classification rows, from
/// [`MakeClassification::stream`].
///
/// As an iterator it yields one `(x, class)` row at a time and never ends;
/// [`next_batch`](Self::next_batch) and [`batches`](Self::batches) draw several
/// rows into one matrix, the same rows however they are grouped.
#[derive(Debug, Clone)]
pub struct ClassificationStream {
    centers: Array2<f64>,
    redundant: Array2<f64>,
    repeated: Vec<usize>,
    n_features: usize,
    flip_y: f64,
    class: WeightedIndex<f64>,
    rng: Xoshiro256PlusPlus,
}

impl ClassificationStream {
    /// The class corners in the informative features, one per row.
    pub fn centers(&self) -> ArrayView2<'_, f64> {
        self.centers.view()
    }

    /// The mix behind the redundant features, `n_informative × n_redundant`.
    pub fn redundant(&self) -> ArrayView2<'_, f64> {
        self.redundant.view()
    }

    /// Which feature each repeated feature copies.
    pub fn repeated(&self) -> &[usize] {
        &self.repeated
    }

    /// The next `n` rows, as one matrix and their classes.
    pub fn next_batch(&mut self, n: usize) -> Samples<usize> {
        let mut x = Array2::zeros((n, self.n_features));
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

    /// Pick a class with the class weights, draw a row of it into `x`, and
    /// return its class after flipping.
    fn next_row(&mut self, x: ArrayViewMut1<'_, f64>) -> usize {
        let class = self.rng.sample(&self.class);
        self.fill_row(class, x)
    }

    /// Draw a row of class `class` into `x`, column kind by column kind, and
    /// return its class after flipping.
    ///
    /// The flip is drawn even when `flip_y` is 0, so a seed gives the same
    /// features at every `flip_y` and only some classes change.
    fn fill_row(&mut self, class: usize, mut x: ArrayViewMut1<'_, f64>) -> usize {
        let (n_informative, n_redundant) = self.redundant.dim();
        let (n_copyable, n_used) = (
            n_informative + n_redundant,
            n_informative + n_redundant + self.repeated.len(),
        );
        for (value, &corner) in x.iter_mut().zip(self.centers.row(class)) {
            let z: f64 = self.rng.sample(StandardNormal);
            *value = corner + z;
        }
        let mixed = x.slice(s![..n_informative]).dot(&self.redundant);
        x.slice_mut(s![n_informative..n_copyable]).assign(&mixed);
        for (offset, &source) in self.repeated.iter().enumerate() {
            x[n_copyable + offset] = x[source];
        }
        for value in x.slice_mut(s![n_used..]) {
            *value = self.rng.sample(StandardNormal);
        }

        let flip = self.rng.random::<f64>() < self.flip_y;
        let random_class = self.rng.random_range(0..self.centers.nrows());
        if flip { random_class } else { class }
    }
}

impl Iterator for ClassificationStream {
    type Item = (Array1<f64>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let mut x = Array1::zeros(self.n_features);
        let class = self.next_row(x.view_mut());
        Some((x, class))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}
