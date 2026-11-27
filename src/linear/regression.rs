//! Linear regression: least squares by SVD, with or without a ridge penalty.

use faer::{Col, Mat};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::dataset::{BlockView, Blocks, Dense, Finite, Orientation, Target, TargetState, Targets};
use crate::error::{check_len, require};
use crate::{Dataset, Fit, Float, Predict, PredictError, Result, ValidateConfig};

const CONTEXT: &str = "LinearRegressor";

/// Linear regression, generic over its penalty `P`: predicts `x · coef +
/// intercept`, choosing `coef` and `intercept` to minimise the squared error
/// on the training rows (weighted by the sample weights, if any), plus the
/// penalty.
///
/// The penalty is a value holding its own settings, so a setting only exists
/// for the penalty that uses it. [`NoPenalty`] is ordinary least squares,
/// named [`LinearRegression`]; [`Ridge`] adds `alpha * ||coef||²`, named
/// [`RidgeRegression`]. The intercept is never penalised.
///
/// With an intercept (the default), features and targets are centred on their
/// (weighted) means before solving, and the intercept is recovered from the
/// means afterwards. See <https://predict.rs/linear.html>.
#[derive(Debug, Clone)]
pub struct LinearRegressor<P = NoPenalty> {
    intercept: bool,
    penalty: P,
}

/// No penalty: ordinary least squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoPenalty;

/// The ridge (L2) penalty, `alpha * ||coef||²`, set through
/// [`RidgeRegression`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ridge {
    alpha: f64,
}

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

/// Ridge regression: [`LinearRegressor`] with the [`Ridge`] penalty, minimising
/// the (weighted) squared error plus `alpha * ||coef||²`, as scikit-learn's
/// `Ridge` does.
///
/// The penalty shrinks every coefficient towards 0, most along the directions
/// in which the features barely vary, so nearly collinear columns get stable,
/// small coefficients instead of large ones that cancel out. It depends on the
/// features' scales, so standardise them first. `alpha = 0` is
/// [`LinearRegression`], exactly.
///
/// Solved by the same SVD as [`LinearRegression`]: each singular value `s`
/// above the same cutoff contributes `1 / (s + alpha / s)` in place of `1 / s`.
/// See <https://predict.rs/linear.html#ridge-regression>.
///
/// Fits the same inputs as [`LinearRegression`].
///
/// # Errors
///
/// From `fit`: `InvalidParameter` (`name = "alpha"`) unless `alpha` is finite,
/// non-negative and within the data's float type, then the errors of
/// [`LinearRegression`].
pub type RidgeRegression = LinearRegressor<Ridge>;

impl LinearRegressor<Ridge> {
    /// Ridge regression with `alpha = 1`, scikit-learn's default, and an
    /// intercept.
    pub fn new() -> Self {
        LinearRegressor {
            intercept: true,
            penalty: Ridge { alpha: 1.0 },
        }
    }

    /// The penalty strength, `>= 0`: larger shrinks the coefficients more.
    pub fn with_alpha(self, alpha: f64) -> Self {
        LinearRegressor {
            penalty: Ridge { alpha },
            ..self
        }
    }

    /// The penalty strength.
    pub fn alpha(&self) -> f64 {
        self.penalty.alpha
    }
}

impl Default for LinearRegressor<Ridge> {
    fn default() -> Self {
        Self::new()
    }
}

/// No settings to check.
impl ValidateConfig for LinearRegressor<NoPenalty> {}

/// `alpha` must be finite and non-negative.
impl ValidateConfig for LinearRegressor<Ridge> {
    fn validate_config(&self) -> Result<()> {
        let alpha = self.penalty.alpha;
        require(
            alpha.is_finite() && alpha >= 0.0,
            "alpha",
            "must be finite and non-negative",
        )
    }
}

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
        least_squares(data, self.intercept, F::zero())
    }
}

impl<'a, F: Float + Target, O: Orientation>
    Fit<Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Targets<'a, F>>>
    for LinearRegressor<Ridge>
{
    type Fitted = FittedLinearRegressor<F>;

    fn fit(
        &self,
        data: &Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Targets<'a, F>>,
    ) -> Result<FittedLinearRegressor<F>> {
        self.validate_config()?;
        // An `f64` alpha can be too large for `f32`.
        let alpha = F::from(self.penalty.alpha).filter(|alpha| alpha.is_finite());
        let Some(alpha) = alpha else {
            return Err(PredictError::InvalidParameter {
                name: "alpha",
                reason: "must be finite in the data's float type".to_string(),
            });
        };
        least_squares(data, self.intercept, alpha)
    }
}

/// The fit both penalties share: minimise the weighted squared error plus
/// `alpha * ||coef||²` (`alpha = 0` without a penalty), the intercept
/// unpenalised.
fn least_squares<'a, F: Float + Target, O: Orientation>(
    data: &Dataset<'a, F, Blocks<'a, F, O, Dense>, Finite, Targets<'a, F>>,
    intercept: bool,
    alpha: F,
) -> Result<FittedLinearRegressor<F>> {
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
    let (x_mean, y_mean) = if intercept {
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

    let coef = minimum_norm_solution(&a, &b, alpha)?;
    let intercept = y_mean - x_mean.dot(&coef);
    Ok(FittedLinearRegressor { coef, intercept })
}

/// The minimiser of `||a · w - b||² + alpha * ||w||²` with the smallest norm,
/// by thin SVD: `w = V D Uᵀ b`, where `D` holds `1 / (s + alpha / s)` for each
/// singular value `s` above the cutoff and zeroes the rest, whose directions
/// are rounding noise. That is `s / (s² + alpha)` without squaring `s`, and
/// exactly `1 / s` when `alpha = 0`.
fn minimum_norm_solution<F: Float>(a: &Mat<F>, b: &Col<F>, alpha: F) -> Result<Array1<F>> {
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
            ut_b[k] / (s[k] + alpha / s[k])
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
