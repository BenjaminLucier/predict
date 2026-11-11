# Synthetic data

Real data never says what the right answer is, so when a model does badly you
can't tell whether the model is wrong or the problem is hard. Synthetic data is
drawn from a process you know, and comes back with that process, so a test can
ask whether a model recovered the truth.

| Generator | Process | Truth returned | For |
|---|---|---|---|
| `MakeRegression` | targets linear in the features, plus noise | coefficients, intercept | linear regression |
| `MakeBlobs` | round clusters around random centres | centres | clustering |
| `MakeClassification` | one cluster per class, on the corners of a cube | corners, feature mixes | classification |

Each generator is a config, like an estimator: `new()`, `with_*` settings, and
`with_seed(n)` to make it reproducible (by default every run differs). Settings
are checked when data is drawn, with an `InvalidParameter` error naming the
first bad one.

## Two ways to draw

`generate()` draws `n_samples` rows and returns them with the truth, in a
struct with public fields. Its `dataset()` method borrows `x` and `y` as a
dataset, validated as finite, with `y` as its targets:

```rust,ignore
{{#rustdoc_include ../../examples/synthetic.rs:regression}}
```

`dataset()` runs the ordinary `validate_finite` scan rather than assuming the
data is finite. Generated features always are, but the fields are public, so
the struct can't promise its values are still the ones generated, and extreme
settings (a `noise` near `f64::MAX`) can overflow a target. The scan costs one
comparison per value, against the random draw each value already cost.

`stream()` returns an endless iterator instead. The truth is drawn first and
readable on the stream; rows are drawn as they're asked for, one at a time or
in batches, each batch a matrix with its own `dataset()`:

```rust,ignore
{{#rustdoc_include ../../examples/synthetic.rs:stream}}
```

A seed fixes the sequence of rows; batching only groups it, so the same seed
gives the same rows however they're drawn. For `MakeRegression`, `generate()`
is exactly the first `n_samples` rows of the stream. The two class generators
differ: `generate()` knows how many rows there will be, so it fixes each
class's size (within one row of its share) and shuffles, while a stream can
only pick each row's class at random, so its class sizes are close to their
shares but not exact.

## Regression

Every feature is drawn from $\mathcal{N}(0, 1)$. The first $k$ of the $p$
features (`n_informative`) get a coefficient drawn uniformly from $(0, 100]$,
the rest get 0, and

$$
y_i = x_i^\top w + b + \sigma \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, 1),
$$

with $b$ the `bias` and $\sigma$ the `noise`. Ordinary least squares on $n$
rows recovers each coefficient with a standard error of about
$\sigma / \sqrt{n}$, because the features are independent with unit variance;
with $\sigma = 0$ it recovers them exactly, up to rounding, which makes a sharp
test. The features with a zero coefficient are there to show what penalised
models do with useless features.

The noise is drawn even when $\sigma = 0$, so a seed gives the same features
and coefficients at every noise level, and only the targets change.

## Blobs

Each centre's coordinates are drawn uniformly from the centre box, $(-10, 10)$
by default, and each row is its centre plus $\mathcal{N}(0, s^2)$ noise on every
feature, with $s$ the `cluster_std`:

```rust,ignore
{{#rustdoc_include ../../examples/synthetic.rs:blobs}}
```

When the centres are far apart compared with $s$, k-means should find centres
near the true ones. As $s$ grows the clusters overlap, and rows in the overlap
can't be assigned to the right cluster by any method.

## Classification

Each class gets its own random corner of a cube in the first `n_informative`
dimensions, every coordinate $\pm c$ with $c$ the `class_sep`, and each row is
its class's corner plus $\mathcal{N}(0, 1)$ noise in every one of those
dimensions:

```rust,ignore
{{#rustdoc_include ../../examples/synthetic.rs:classification}}
```

Corners make the difficulty something you can work out. Take two classes whose
corners differ on $m$ axes. They are $d = 2c\sqrt{m}$ apart, and since the noise
is round with unit variance, the best possible classifier splits the space
halfway between them. Projected onto the line joining the corners, each class
is a unit normal centred $d/2$ from the split, so the best classifier is wrong
with probability

$$
\Phi\!\left(-\frac{d}{2}\right) = \Phi\!\left(-c\sqrt{m}\right),
$$

where $\Phi$ is the standard normal distribution function. At $c = 1$ and
$m = 1$ that's about 16%; at $c = 2$, about 2.3%. A model's accuracy can be
compared with that bound. There are $2^k$ corners in $k$ informative
dimensions, so `n_classes` can be at most $2^k$.

After the informative features come three more kinds, in this order:

- **Redundant** features (`n_redundant`): each is the informative features
  times a random weight vector, weights uniform in $[-1, 1)$. They carry no new
  information but are correlated with the informative ones, which is what makes
  linear models' coefficients unstable.
- **Repeated** features (`n_repeated`): exact copies of random informative or
  redundant features.
- **Useless** features, the rest: $\mathcal{N}(0, 1)$ noise, unrelated to the
  class.

Last, with probability `flip_y` each row's class is replaced by a class picked
at random (possibly the same one), so even the best classifier can't be
perfect, as with real data. The flip is drawn whatever `flip_y` is, so a seed
gives the same features at every `flip_y`.

The defaults are scikit-learn's: 20 features (2 informative, 2 redundant), 2
balanced classes, $c = 1$, `flip_y` 0.01. scikit-learn also places 2 clusters
per class by default, stretches each cluster with a random covariance, and
shuffles the columns; this crate doesn't yet.
