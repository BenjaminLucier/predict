# Errors

Every fallible function in this crate returns `predict::Result<T>`, whose error
is the single enum `predict::PredictError`. One type for the whole crate means a
pipeline can pass on an error from any of its steps with `?`, and callers can
`match` on the kind of failure rather than parse a message.

The aim is to fail early and say exactly where. A NaN that reaches a matrix
product spreads into every coefficient and gives a model that predicts nonsense
without complaint, so inputs are checked before any work and the error names the
first non-finite value in memory order:
`non-finite value NaN in y at row 41, column 0`.

## The variants

| Variant | Meaning |
|---|---|
| `NonFinite { context, row, col, value }` | A NaN or infinity where finite values are required. `row` and `col` give the first non-finite value in memory order; with several, which one is reported depends on how the array is stored. A 1-D input reports `col = 0`. |
| `ShapeMismatch { context, expected, actual }` | A length or column count is not the one required, such as targets with a different number of rows than the features. |
| `EmptyInput { context }` | An input has no rows or no columns where at least one is required. |
| `InvalidParameter { name, reason }` | A setting is outside its valid range, or conflicts with the data. |

`PredictError` is `#[non_exhaustive]`: match it with a `_ =>` arm, so new
variants can be added without breaking your code. Its variants can be
constructed, so an estimator written outside the crate can return them too.

It implements `std::error::Error` and is `Send + Sync + 'static`, so it works with
`?` in functions returning `anyhow::Error` or `Box<dyn Error + Send + Sync>`, and
crosses threads. It does not implement `Clone` or `PartialEq`; compare errors with
`matches!(err, PredictError::ShapeMismatch { .. })`.

## Reading an error

`context` says where an error came from, in one of three forms:

- an input, by its argument name: `"x"`, `"y"`, `"y_true"`;
- a transformer, by its type name, from `fit` and `transform` alike:
  `"StandardScaler"`;
- the method, where neither applies: `"validate_finite"`.

`InvalidParameter`'s `name` is the setting, as in the method that sets it
(`max_iter` for `with_max_iter`). Its `reason` states the requirement, such as
`must be at least 1`, or the conflict when the data is involved, adding
`, got {value}` where the value is not obvious.

Settings are checked when they are used, in `fit`, not when they are set: the
`with_*` methods only store a value, so a configuration chain never returns a
`Result`.
