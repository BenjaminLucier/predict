//! An estimator written against the traits: a nearest-centroid classifier, as
//! the book's Architecture page walks through. Run with
//! `cargo run --example estimator`.
//!
//! Nearest centroid learns one point per class, the weighted mean of that
//! class's rows, and predicts the class whose point is closest. It is
//! scikit-learn's `NearestCentroid` without its shrinkage setting.

use ndarray::{Array1, Array2, ArrayView2, array, s};
use predict::dataset::{BlockView, DenseRowMajor, Finite, State, TargetState, Targets};
use predict::{Dataset, Fit, Predict, PredictError, Result, ValidateConfig};

// ANCHOR: config
/// The config: the settings a user chooses. Nearest centroid has none.
#[derive(Debug, Clone, Default)]
pub struct NearestCentroid;

/// What fitting learns: one centroid per class, classes × features.
#[derive(Debug, Clone)]
pub struct FittedNearestCentroid {
    centroids: Array2<f64>,
}

/// No settings, so nothing to check: the default accepts.
impl ValidateConfig for NearestCentroid {}
// ANCHOR_END: config

// ANCHOR: fit
/// Fits on finite, row-major, dense features with class indices as targets.
/// Any other dataset is a compile error, not a runtime one.
impl<'a> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, usize>>> for NearestCentroid {
    type Fitted = FittedNearestCentroid;

    fn fit(
        &self,
        data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, usize>>,
    ) -> Result<FittedNearestCentroid> {
        self.validate_config()?;
        let y = data.y();
        let n_classes = y.iter().max().map_or(0, |&class| class + 1);
        let mut sums = Array2::zeros((n_classes, data.n_features()));
        let mut weights = Array1::<f64>::zeros(n_classes);
        for (row, &class) in y.iter().enumerate() {
            let w = data.sample_weights().map_or(1.0, |w| w[row]);
            let mut start = 0;
            for block in rows(data) {
                let end = start + block.ncols();
                sums.slice_mut(s![class, start..end])
                    .scaled_add(w, &block.row(row));
                start = end;
            }
            weights[class] += w;
        }
        if weights.iter().any(|&w| w <= 0.0) {
            return Err(PredictError::EmptyInput {
                context: "a class with no weight",
            });
        }
        let centroids = sums / weights.insert_axis(ndarray::Axis(1));
        Ok(FittedNearestCentroid { centroids })
    }
}
// ANCHOR_END: fit

// ANCHOR: predict
/// Predicts on the same layout and state, with or without targets: the impl is
/// generic over `Y`, so it can't read the answers.
impl<'a, Y: TargetState> Predict<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>>
    for FittedNearestCentroid
{
    type Output = Array1<usize>;

    fn predict(&self, x: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<Array1<usize>> {
        let n_features = self.centroids.ncols();
        if x.n_features() != n_features {
            return Err(PredictError::ShapeMismatch {
                context: "x",
                expected: n_features,
                actual: x.n_features(),
            });
        }
        let blocks: Vec<ArrayView2<'_, f64>> = rows(x).collect();
        let predictions = (0..x.n_samples())
            .map(|row| {
                let distance = |class: usize| {
                    let mut start = 0;
                    let mut total = 0.0;
                    for block in &blocks {
                        let end = start + block.ncols();
                        let centroid = self.centroids.slice(s![class, start..end]);
                        total += (&block.row(row) - &centroid).mapv(|d| d * d).sum();
                        start = end;
                    }
                    total
                };
                (0..self.centroids.nrows())
                    .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
                    .unwrap_or(0)
            })
            .collect();
        Ok(predictions)
    }
}
// ANCHOR_END: predict

/// The dense row-major blocks of a `DenseRowMajor` dataset, borrowed.
fn rows<'b, 'a, S: State, Y: TargetState>(
    data: &'b Dataset<'a, f64, DenseRowMajor<'a>, S, Y>,
) -> impl Iterator<Item = ArrayView2<'b, f64>> {
    data.blocks().map(|block| match block {
        BlockView::DenseRowMajor(x) => x,
        _ => unreachable!("a DenseRowMajor dataset holds only row-major dense blocks"),
    })
}

fn main() -> Result<()> {
    // ANCHOR: use
    // Two clusters: class 0 near the origin, class 1 near (10, 10).
    let x = array![[0.0, 1.0], [1.0, 0.0], [10.0, 9.0], [9.0, 10.0]];
    let y = array![0_usize, 0, 1, 1];
    let data = Dataset::new(&x).validate_finite()?.with_y(&y)?;

    let model = NearestCentroid.fit(&data)?;
    let new = array![[0.5, 0.5], [8.0, 8.0]];
    let predictions = model.predict(&Dataset::new(&new).validate_finite()?)?;
    assert_eq!(predictions, array![0, 1]);
    // ANCHOR_END: use
    println!(
        "centroids:\n{}\npredictions: {predictions}",
        model.centroids
    );
    Ok(())
}
