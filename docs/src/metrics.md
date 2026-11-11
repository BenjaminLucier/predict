# Metrics

A metric turns a model's predictions into one number that says how good they
are. Every metric here is a free function in `predict::metrics` that takes the
true targets `y_true` and the predictions `y_pred` as 1-D array views and
returns an `f64`.

```rust
{{#rustdoc_include ../../examples/metrics.rs:regression}}
```

## Inputs and checks

The output of the `predict` method isn't validated (a model can produce
infinity), so every metric checks its inputs before computing anything:

| Problem | Error |
|---|---|
| `y_pred` has a different length from `y_true` | `ShapeMismatch`, `context = "y_pred"` |
| no rows | `EmptyInput`, `context = "y_true"` |
| a NaN or infinity in `y_true` or `y_pred` | `NonFinite`, at its row |

Inputs can be `f32` or `f64`, but every result is an `f64`, and the sums behind
it are taken in `f64` whatever the input type. A score is read and compared,
never fed back into a model's arithmetic, so the wider type costs nothing. It
keeps sums of `f32` values accurate, and an `f32` difference squared always
fits in an `f64`, so `f32` inputs can't overflow.

## Regression

The residual of a row is $y_i - \hat y_i$, how far its prediction is from the
truth.

| Function | Formula | Best |
|---|---|---|
| `mse` | $\frac{1}{n}\sum_i (y_i - \hat y_i)^2$ | 0 |
| `rmse` | $\sqrt{\text{MSE}}$ | 0 |
| `mae` | $\frac{1}{n}\sum_i \lvert y_i - \hat y_i \rvert$ | 0 |
| `r2_score` | $1 - SS_{res} / SS_{tot}$ | 1 |

**MSE** squares each residual, so one error of 10 counts as much as a hundred
errors of 1: it punishes large mistakes heavily, and one outlier can dominate
it. It's in the targets' units squared; **RMSE** takes the square root to get
back to the targets' units. **MAE** counts every unit of error the same, so
outliers move it far less.

**R²**, the coefficient of determination, compares the model with the most
naive one, which always predicts the mean $\bar y$:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

1 is perfect, 0 is no better than the mean, and it goes negative, without a
lower bound, when the model is worse than the mean. It has no units, so unlike
the others it compares across problems.

When `y_true` is constant, including a single row, the denominator is zero:
there is no variance to explain, and `r2_score` returns `Undefined`.
scikit-learn returns 1.0 or 0.0 there instead, which can quietly inflate the
scores of small cross-validation folds. A sum of squares outside the range of
`f64` is `Undefined` too.

Errors too large for `f64` give infinity from `mse`, `rmse` and `mae`, and
negative infinity from `r2_score`, never NaN: hopeless predictions score worst
rather than stopping an evaluation.

## Sample weights

Each metric has a `_weighted` form that takes one weight per row and counts row
$i$ $w_i$ times: a weight of 2 is the same as repeating the row, and a weight of
0 leaves it out. The mean becomes a weighted mean,
$\sum_i w_i t_i / \sum_i w_i$ for each row's term $t_i$; in R², the mean of
`y_true` is weighted too.

```rust
{{#rustdoc_include ../../examples/metrics.rs:weighted}}
```

Weights are checked as `Dataset` checks them (one per row, finite,
non-negative), and their total must be positive and finite, or the weighted
mean has no value (`Undefined`). Weights can be a different float type from the
targets, as they can in a `Dataset`.

## Classification

Classes are `usize` indices, as in targets. **Accuracy** is the fraction of
rows whose predicted class is the true one. On imbalanced classes it flatters:
always predicting the majority class of a 95 / 5 split scores 0.95 while never
finding a minority row.

A **confusion matrix** counts every (true class, predicted class) pair; the
diagonal holds the correct predictions. Its classes are those that appear in
`y_true` or `y_pred`, sorted.

```rust
{{#rustdoc_include ../../examples/metrics.rs:classification}}
```

For each class, with TP its true positives, FP the rows wrongly predicted as it
and FN its rows predicted as something else:

- **precision** $P = TP / (TP + FP)$: when the model says this class, how often
  is it right?
- **recall** $R = TP / (TP + FN)$: of this class's rows, how many does it find?
- **F-beta**, a weighted harmonic mean of the two:

$$F_\beta = \frac{(1 + \beta^2)\,P\,R}{\beta^2 P + R}$$

A harmonic mean is pulled towards the smaller value, so a model can't score
well by maximising one at the expense of the other. $\beta > 1$ counts recall
more, $\beta < 1$ precision more, and `f1_score` is $\beta = 1$.

A class that is never predicted has precision $0/0$; this crate scores it 0, as
scikit-learn does by default, and keeps it in the average. A class the model
never predicts really did get no value from it, and an error would stop every
cross-validation fold that misses a rare class. Recall and F-beta follow the
same rule.

### Combining classes

`Average` says how the per-class scores become one number:

| `Average` | Meaning |
|---|---|
| `Binary { positive }` | the score of the `positive` class alone, for two-class problems |
| `Macro` | the plain mean over classes; a rare class counts as much as a common one |
| `Micro` | one score from the counts summed over all classes; with one class per row it equals accuracy |
| `BySupport` | the mean weighted by each class's support, its number of true rows |

Most two-class problems have one class of interest (fraud, disease, spam), and
"the precision" means the precision of that class; `Binary` returns it. Unlike
scikit-learn, where `"binary"` is the default with class 1 as positive, the
positive class is always named. scikit-learn calls `BySupport` `"weighted"`;
the name here keeps it apart from the sample weights of the `_weighted`
functions.

## Metrics as values

Code that evaluates models needs a metric it can be given and call, and it
needs to know which way is better. The `Metric` trait provides both:
`compute(y_true, y_pred, sample_weights)` calls the matching function, and
`direction()` returns a `Direction`. Each metric has a type: `MSE`, `RMSE`,
`MAE`, `R2`, `Accuracy`, and `Precision`, `Recall`, `F1` and `FBeta`, which
carry their `Average` (and `beta`).

```rust
{{#rustdoc_include ../../examples/metrics.rs:metric}}
```

scikit-learn instead makes every score "higher is better" by negating the
errors, which is why its users meet `neg_mean_squared_error`. Here an error
stays positive and says which way it points.

The trait is open, so a metric of your own is a type that implements it.
