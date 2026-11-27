# predict

This crate is machine learning for Rust with a consistent, scikit-learn-style
API.

**Polars to production.** It picks up where a dataframe library such as
Polars hands off, at plain `ndarray` arrays, and aims to cover everything from
there to a model running in production, in Rust, for the algorithms
practitioners actually reach for. Five things guide it:

- **Speed:** traditional ML as fast as a single machine allows.
- **Correctness:** results checked against scikit-learn, every input
  validated, and errors instead of silent NaN.
- **Type safety:** a dataset's checks are part of its type, so mistakes such
  as fitting unvalidated data fail to compile.
- **Rust-native:** pure Rust, with no Python runtime and no C or BLAS
  libraries to install.
- **Batteries included:** the pieces around a model ship with it, from
  splitting and cross-validation to metrics.

It is selective on purpose: an algorithm is included because people use it in
practice, and each one is done properly rather than added for the count.

It is in early development, and every `0.0.x` release may break the API. This
book grows with the code. Each page describes what is in the repository today:
datasets, synthetic data, metrics, model selection and linear regression
(least squares and ridge).
