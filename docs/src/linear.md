# Linear models

A linear model predicts a target as a weighted sum of the features plus an
intercept, $\hat y = Xw + b$. It is fast to fit, easy to read (each coefficient
says how much the prediction moves per unit of its feature), and the baseline
most other models are compared with.

`predict::linear` has one estimator, `LinearRegressor<P>`, generic over its
penalty `P`. This page covers `LinearRegression`, which is `LinearRegressor`
with no penalty: ordinary least squares.

```rust,ignore
{{#rustdoc_include ../../examples/linear.rs:fit}}
```

## Ordinary least squares

Least squares picks the coefficients and intercept that minimise the squared
error over the training rows, weighted by the sample weights $s_i$ (all 1
without weights):

$$
\min_{w,\,b} \; \sum_i s_i \left(y_i - x_i^\top w - b\right)^2
$$

Geometrically, $Xw$ ranges over the combinations of $X$'s columns, a flat
subspace, and the best fit is the point of it closest to $y$: the projection of
$y$ onto it. There the residual $y - Xw$ is perpendicular to every column,
which gives the *normal equations* $X^\top X w = X^\top y$.

Least squares fails on curved relationships (unless features such as squares
are added), and squared error lets one far-off row pull the whole fit.

### The intercept

For any $w$, the best intercept is $b = \bar y - \bar x^\top w$, with $\bar x$ and
$\bar y$ the (weighted) means. So fitting with an intercept is fitting without
one on centred data: subtract the means, solve, and recover $b$ from them.
`without_intercept()` skips the centring, for a fit through the origin or
features that already hold a column of ones.

### Solving by SVD

Forming $X^\top X$ squares the condition number of $X$, how much it amplifies
small errors: a fairly ill-conditioned $10^8$ becomes $10^{16}$, every digit an
`f64` has. So the fit never forms it. It copies the centred features, each row
scaled by $\sqrt{s_i}$, into a working matrix once, and takes its singular value
decomposition $X = U \Sigma V^\top$:

$$
w = V \, \Sigma^{+} \, U^\top y
$$

$\Sigma^{+}$ inverts each singular value $\sigma_k$ above
$\max(n, p) \cdot \varepsilon \cdot \sigma_{\max}$ and zeroes the rest, as NumPy's
`lstsq` does: directions that small are rounding noise, and dividing by them
would turn noise into huge coefficients.

### Many best fits

When the columns are linearly dependent, many coefficient vectors fit equally
well: a duplicated column, one-hot categories plus an intercept, or more
features than rows. Zeroing the smallest singular values picks the one with the
smallest $\lVert w \rVert$, the *minimum-norm* solution, which is what
scikit-learn's `LinearRegression` returns. A duplicated column gets half the
coefficient on each copy.

## Inputs

`fit` takes finite, dense features in any orientation (row-major, column-major
or both side by side), since the working matrix is a copy either way, with
targets of the same float type. Sparse features don't compile. The fit computes
in the data's float type: `f32` keeps about 7 digits, so ill-conditioned
features want `f64`.

| Error | When |
|---|---|
| `EmptyInput` | no rows |
| `Undefined` | the sample weights add up to 0, or the SVD doesn't converge |
| `ShapeMismatch`, `context = "x"` | `predict` gets a different number of features |

## Cross-validated

On data drawn with a little noise, every fold scores an $R^2$ near 1, against
the mean baseline's 0 on the [model selection](selection.md) page:

```rust,ignore
{{#rustdoc_include ../../examples/linear.rs:cross_validation}}
```
