//! Linear models: a prediction that is a weighted sum of the features plus an
//! intercept. See <https://predict.rs/linear.html>.
//!
//! [`LinearRegressor<P>`] is one estimator generic over its penalty `P`, a
//! value holding the penalty's own settings. [`NoPenalty`] gives ordinary
//! least squares, named [`LinearRegression`]. Every penalty learns the same
//! two things, so all of them fit a [`FittedLinearRegressor`].

mod regression;

pub use regression::{
    FittedLinearRegressor, LinearRegression, LinearRegressor, NoPenalty, Ridge, RidgeRegression,
};
