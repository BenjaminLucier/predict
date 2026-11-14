//! Linear regression: least squares by SVD.

use faer::{Col, Mat};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::dataset::{BlockView, Blocks, Dense, Finite, Orientation, Target, TargetState, Targets};
use crate::error::check_len;
use crate::{Dataset, Fit, Float, Predict, PredictError, Result, ValidateConfig};

const CONTEXT: &str = "LinearRegressor";

/// Linear regression, generic over its penalty `P`: predicts `x · coef +
/// intercept`, choosing `coef` and `intercept` to minimise the squared error
/// on the training rows (weighted by the sample weights, if any), plus the
/// penalty.
///
/// The penalty is a value holding its own settings, so a setting only exists
/// for the penalty that uses it. [`NoPenalty`] is ordinary least squares,
/// named [`LinearRegression`].
///
/// With an intercept (the default), features and targets are centred on their
/// (weighted) means before solving, and the intercept is recovered from the
/// means afterwards. See <https://predict.rs/linear.html>.
#[derive(Debug, Clone)]
pub struct LinearRegressor<P = NoPenalty> {
    intercept: bool,
    #[expect(dead_code, reason = "NoPenalty has no settings to read")]
    penalty: P,
}

/// No penalty: ordinary least squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoPenalty;

/// Ordinary least squares: [`LinearRegressor`] with [`NoPenalty`].
///
/// Solved by the SVD of the centred, weighted features, copied once into a
/// working matrix. Singular values below `max(n_samples, n_features) * ε *
/// s_max` count as zero, as in NumPy's `lstsq`, so linearly dependent columns
/// (a duplicated column, one-hot columns with an intercept) and more features
/// than rows give the minimum-norm solution, as scikit-learn's
/// `LinearRegression` does.
///
/// It computes in the data's float type: `f32` keeps about 7 digits, so with
/// ill-conditioned features use `f64`.
///
/// Fits finite, dense features in any orientation, with targets of the same
/// float type. Sparse features don't compile:
///
/// ```compile_fail
/// use predict::Fit;
/// use predict::dataset::CsrMatrix;
/// use predict::linear::LinearRegression;
/// use predict::Dataset;
///
/// let x = CsrMatrix::new(1, vec![0, 1], vec![0], vec![1.0]).unwrap();
/// let y = ndarray::array![1.0];
/// let data = Dataset::from_csr(x).validate_finite().unwrap().with_y(&y).unwrap();
/// LinearRegression::new().fit(&data); // sparse
/// ```
///
/// # Errors
///
/// From `fit`: `EmptyInput` with no rows, `Undefined` if the sample weights
/// add up to 0 or the SVD doesn't converge.
pub type LinearRegression = LinearRegressor<NoPenalty>;

impl LinearRegressor<NoPenalty> {
    /// Ordinary least squares, with an intercept.
    pub fn new() -> Self {
        LinearRegressor {
            intercept: true,
            penalty: NoPenalty,
        }
    }
}

impl Default for LinearRegressor<NoPenalty> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> LinearRegressor<P> {
    /// Fit no intercept: the prediction for all-zero features is 0. For when
    /// the relationship must pass through the origin, or the features already
    /// hold a column of ones. Nothing is centred.
    pub fn without_intercept(self) -> Self {
        LinearRegressor {
            intercept: false,
            ..self
        }
    }
}

/// No settings to check.
impl ValidateConfig for LinearRegressor<NoPenalty> {}

impl<'a, F: Float + Target, O: Orientation>
    Fit<Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Targets<'a, F>>>
    for LinearRegressor<NoPenalty>
{
    type Fitted = FittedLinearRegressor<F>;

    fn fit(
        &self,
        data: &Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Targets<'a, F>>,
    ) -> Result<FittedLinearRegressor<F>> {
        self.validate_config()?;
        let n_samples = data.n_samples();
        if n_samples == 0 {
            return Err(PredictError::EmptyInput { context: CONTEXT });
        }
        let weights = data.sample_weights();
        let total = weights.map_or(F::from(n_samples).unwrap(), |w| w.sum());
        if total <= F::zero() {
            return Err(PredictError::Undefined {
                context: CONTEXT,
                reason: "the sample weights add up to 0".to_string(),
            });
        }

        let blocks: Vec<ArrayView2<'_, F>> = data.blocks().map(dense).collect();
        let y = data.y();
        let (x_mean, y_mean) = if self.intercept {
            (
                column_means(&blocks, weights, total),
                mean(y, weights, total),
            )
        } else {
            (Array1::zeros(data.n_features()), F::zero())
        };

        // The working matrix: centred, each row scaled by the square root of
        // its weight, so the plain least squares solution of `a · coef = b`
        // minimises the weighted squared error.
        let row_scale = |row: usize| weights.map_or(F::one(), |w| w[row].sqrt());
        let mut a = Mat::<F>::zeros(n_samples, data.n_features());
        let mut start = 0;
        for block in &blocks {
            for (j, column) in block.columns().into_iter().enumerate() {
                let (col, centre) = (start + j, x_mean[start + j]);
                for (row, &value) in column.iter().enumerate() {
                    a[(row, col)] = (value - centre) * row_scale(row);
                }
            }
            start += block.ncols();
        }
        let b = Col::<F>::from_fn(n_samples, |row| (y[row] - y_mean) * row_scale(row));

        let coef = minimum_norm_solution(&a, &b)?;
        let intercept = y_mean - x_mean.dot(&coef);
        Ok(FittedLinearRegressor { coef, intercept })
    }
}

/// The least squares solution of `a · w = b` with the smallest norm, by thin
/// SVD: `w = V Σ⁺ Uᵀ b`, where `Σ⁺` inverts each singular value above the
/// cutoff and zeroes the rest, whose directions are rounding noise.
fn minimum_norm_solution<F: Float>(a: &Mat<F>, b: &Col<F>) -> Result<Array1<F>> {
    let svd = a.thin_svd().map_err(|_| PredictError::Undefined {
        context: CONTEXT,
        reason: "the SVD of the features did not converge".to_string(),
    })?;
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let (n_rows, n_cols) = a.shape();
    // Singular values come largest first.
    let cutoff = if s.nrows() == 0 {
        F::zero()
    } else {
        F::from(n_rows.max(n_cols)).unwrap() * F::epsilon() * s[0]
    };
    let ut_b = u.transpose() * b;
    let scaled = Col::<F>::from_fn(s.nrows(), |k| {
        if s[k] > cutoff {
            ut_b[k] / s[k]
        } else {
            F::zero()
        }
    });
    let w = v * scaled;
    Ok((0..n_cols).map(|j| w[j]).collect())
}

/// Each column's mean, weighted when there are weights.
fn column_means<F: Float>(
    blocks: &[ArrayView2<'_, F>],
    weights: Option<ArrayView1<'_, F>>,
    total: F,
) -> Array1<F> {
    let parts: Vec<Array1<F>> = blocks
        .iter()
        .map(|block| match weights {
            Some(w) => block.t().dot(&w) / total,
            None => block.sum_axis(Axis(0)) / total,
        })
        .collect();
    let views: Vec<ArrayView1<'_, F>> = parts.iter().map(|part| part.view()).collect();
    ndarray::concatenate(Axis(0), &views).expect("1-D parts always concatenate")
}

fn mean<F: Float>(y: ArrayView1<'_, F>, weights: Option<ArrayView1<'_, F>>, total: F) -> F {
    match weights {
        Some(w) => y.dot(&w) / total,
        None => y.sum() / total,
    }
}

/// A dense block's values: the only kind a `Dense` layout holds.
fn dense<F: Float>(block: BlockView<'_, F>) -> ArrayView2<'_, F> {
    match block {
        BlockView::DenseRowMajor(x) | BlockView::DenseColMajor(x) => x,
        _ => unreachable!("a Dense layout holds only dense blocks"),
    }
}

/// What a [`LinearRegressor`] learns: a coefficient per feature and an
/// intercept. Predicts `x · coef + intercept`.
#[derive(Debug, Clone)]
pub struct FittedLinearRegressor<F: Float = f64> {
    coef: Array1<F>,
    intercept: F,
}

impl<F: Float> FittedLinearRegressor<F> {
    /// One coefficient per feature, in column order.
    pub fn coef(&self) -> ArrayView1<'_, F> {
        self.coef.view()
    }

    /// The prediction for all-zero features: 0 when fitted without an
    /// intercept.
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

/// Predicts on finite, dense features of the same width, with or without
/// targets.
impl<'a, F: Float, O: Orientation, Y: TargetState>
    Predict<Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Y>> for FittedLinearRegressor<F>
{
    type Output = Array1<F>;

    fn predict(&self, x: &Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Y>) -> Result<Array1<F>> {
        check_len("x", self.coef.len(), x.n_features())?;
        let mut y_pred = Array1::from_elem(x.n_samples(), self.intercept);
        let mut start = 0;
        for block in x.blocks().map(dense) {
            let end = start + block.ncols();
            y_pred += &block.dot(&self.coef.slice(ndarray::s![start..end]));
            start = end;
        }
        Ok(y_pred)
    }
}
